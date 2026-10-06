#include "mod-ollama-chat_director.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat_events.h"
#include "mod-ollama-chat_world.h"
#include "mod-ollama-chat-utilities.h"
#include "CellImpl.h"
#include "Creature.h"
#include "DatabaseEnv.h"
#include "GameTime.h"
#include "GridNotifiers.h"
#include "GridNotifiersImpl.h"
#include "Log.h"
#include "Map.h"
#include "ObjectMgr.h"
#include "Player.h"
#include "Random.h"
#include "World.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// mod-ledger records what the DM actually put into the world, so dm.py can see a line was delivered and the
// loop closes on the ledger (plan 32 section 1). Weak, so this module still links without mod-ledger. Keep
// the signature in step with mod_ledger_scripts.cpp.
void LedgerRecordScene(Player* witness, Creature* speaker, char const* kind, uint64 lineId) __attribute__((weak));

namespace
{
    struct DirectorLine
    {
        uint64      id;
        std::string words;
    };

    using LineMap = std::unordered_map<uint64, std::vector<DirectorLine>>;

    struct DirectorLines
    {
        // (instance id << 32 | speaker entry) -> lines, best first. Instance 0 holds the lines written for
        // anyone (dm.py boss-words), which are the fallback when nothing was written for this party.
        LineMap boss;
        // (player guid << 32 | zone id) -> lines, best first. Any innkeeper in the zone may say them: what is
        // said is the zone's talk and the player's standing in it, not the innkeeper's own story.
        LineMap inn;
    };

    std::mutex                            g_DirectorMutex;
    std::shared_ptr<const DirectorLines>  g_DirectorLines;
    std::atomic<bool>               g_DirectorLoading{ false };

    // (instance id << 32 | entry) of every boss that has already spoken. Instance maps update on their own
    // threads, so this is shared between them and needs the lock.
    std::unordered_set<uint64>      g_Spoken;

    // (map id << 8 | difficulty) -> the entries of that dungeon's final encounters. Usually one; Scarlet
    // Monastery has four, one per wing, and each wing is its own instance.
    std::unordered_map<uint32, std::vector<uint32>> g_FinalBosses;

    // Per real player: ms since the last approach check.
    std::unordered_map<ObjectGuid::LowType, uint32> g_CheckTimers;

    // Inn lines already said (by dm_line id), and when each player was last spoken to by an innkeeper. Both
    // are lost at a restart; dm.py does not hand out a line it has seen spoken (dm_scene), so the worst case
    // is one innkeeper speaking again early.
    std::unordered_set<uint64>                      g_InnSaid;
    std::unordered_map<ObjectGuid::LowType, time_t> g_InnLastSpoke;

    uint64 Key(uint32 instanceId, uint32 entry)
    {
        return (uint64(instanceId) << 32) | entry;
    }

    bool TableExists(char const* name)
    {
        return bool(CharacterDatabase.Query(SafeFormat(
            "SELECT 1 FROM information_schema.tables WHERE table_schema = DATABASE() AND table_name = '{}'", name)));
    }

    void LoadLines()
    {
        if (!TableExists("dm_line"))
            return;

        auto lines = std::make_shared<DirectorLines>();
        if (QueryResult result = CharacterDatabase.Query(
                "SELECT id, instance_id, speaker_entry, words FROM dm_line "
                "WHERE scene = 'boss_approach' AND (expires_at IS NULL OR expires_at > NOW()) "
                "ORDER BY instance_id, speaker_entry, `rank`, id"))
        {
            do
            {
                Field* f = result->Fetch();
                lines->boss[Key(f[1].Get<uint32>(), f[2].Get<uint32>())].push_back(
                    { f[0].Get<uint64>(), f[3].Get<std::string>() });
            } while (result->NextRow());
        }

        if (g_DirectorInnScene)
            if (QueryResult result = CharacterDatabase.Query(
                    "SELECT id, for_guid, zone_id, words FROM dm_line "
                    "WHERE scene = 'inn_gossip' AND for_guid <> 0 AND (expires_at IS NULL OR expires_at > NOW()) "
                    "ORDER BY for_guid, zone_id, `rank`, id"))
            {
                do
                {
                    Field* f = result->Fetch();
                    lines->inn[Key(f[1].Get<uint32>(), f[2].Get<uint32>())].push_back(
                        { f[0].Get<uint64>(), f[3].Get<std::string>() });
                } while (result->NextRow());
            }

        std::lock_guard<std::mutex> lock(g_DirectorMutex);
        g_DirectorLines = std::move(lines);
    }

    std::shared_ptr<const DirectorLines> Lines()
    {
        std::lock_guard<std::mutex> lock(g_DirectorMutex);
        return g_DirectorLines;
    }

    // The final encounter of a dungeon is the one DungeonEncounter.dbc marks with the LFG dungeon it ends
    // (instance_encounters.lastEncounterDungeon). Creature::IsDungeonBoss() is true of every encounter boss,
    // so it cannot tell the head boss from the first one.
    std::vector<uint32> FinalBosses(Map* map)
    {
        uint32 const key = (map->GetId() << 8) | uint32(map->GetDifficulty());

        std::lock_guard<std::mutex> lock(g_DirectorMutex);
        auto it = g_FinalBosses.find(key);
        if (it != g_FinalBosses.end())
            return it->second;

        std::vector<uint32> entries;
        if (DungeonEncounterList const* encounters = sObjectMgr->GetDungeonEncounterList(map->GetId(), map->GetDifficulty()))
            for (DungeonEncounter const* encounter : *encounters)
                if (encounter->creditType == ENCOUNTER_CREDIT_KILL_CREATURE && encounter->lastEncounterDungeon)
                    entries.push_back(encounter->creditEntry);

        g_FinalBosses.emplace(key, entries);
        return entries;
    }

    // Instance first, then the lines for anyone. The party's own line is the best one dm.py wrote; a
    // fallback is drawn at random so two runs of the same dungeon need not hear the same words.
    DirectorLine const* PickLine(LineMap const& lines, uint32 instanceId, uint32 entry)
    {
        auto it = lines.find(Key(instanceId, entry));
        if (it != lines.end() && !it->second.empty())
            return &it->second.front();

        if (!g_DirectorBossFallback)
            return nullptr;

        it = lines.find(Key(0, entry));
        if (it != lines.end() && !it->second.empty())
            return &it->second[urand(0, uint32(it->second.size()) - 1)];

        return nullptr;
    }

    void CheckApproach(Player* player)
    {
        Map* map = player->GetMap();
        if (!map || !map->IsDungeon() || map->IsRaid() || !player->IsAlive())
            return;

        uint32 const instanceId = map->GetInstanceId();
        auto lines = Lines();
        if (!lines)
            return;

        for (uint32 entry : FinalBosses(map))
        {
            uint64 const key = Key(instanceId, entry);
            {
                std::lock_guard<std::mutex> lock(g_DirectorMutex);
                if (g_Spoken.count(key))
                    continue;
            }

            Creature* boss = player->FindNearestCreature(entry, float(g_DirectorBossRange), true);
            if (!boss || boss->IsInCombat() || boss->IsInEvadeMode() || !player->IsWithinLOSInMap(boss))
                continue;

            DirectorLine const* line = PickLine(lines->boss, instanceId, entry);
            if (!line)
                continue;

            {
                std::lock_guard<std::mutex> lock(g_DirectorMutex);
                if (!g_Spoken.insert(key).second)
                    continue;
            }

            // Universal, so a boss of either side is understood: the line is written for the party, and an
            // orc's taunt that reads as gibberish to a human is not a scene.
            boss->Yell(line->words, LANG_UNIVERSAL, player);

            if (LedgerRecordScene)
                LedgerRecordScene(player, boss, "boss_approach", line->id);

            OllamaEvents_SceneSpoken(player, boss->GetName(), line->words,
                                     "{0} called out to us before the fight: \"{1}\"", g_DirectorAnswerEventType, true);

            if (g_DebugEnabled)
                LOG_INFO("server.loading", "[Ollama Chat] Director: {} spoke to {} (instance {}, line {})",
                    boss->GetName(), player->GetName(), instanceId, line->id);
        }
    }

    // The nearest living innkeeper within range. A grid visit of a few yards, once a second, for real players
    // only: the same search FindNearestCreature makes, with the innkeeper flag in place of an entry.
    class NearestInnkeeperCheck
    {
    public:
        NearestInnkeeperCheck(WorldObject const& obj, float range) : _obj(obj), _range(range) { }

        bool operator()(Creature* creature)
        {
            if (creature->IsAlive() && creature->HasNpcFlag(UNIT_NPC_FLAG_INNKEEPER) && _obj.IsWithinDistInMap(creature, _range))
            {
                _range = _obj.GetDistance(creature);
                return true;
            }
            return false;
        }

    private:
        WorldObject const& _obj;
        float _range;
    };

    void CheckInn(Player* player)
    {
        if (!player->IsAlive() || player->IsInCombat())
            return;

        ObjectGuid::LowType const guid = player->GetGUID().GetCounter();
        time_t const now = GameTime::GetGameTime().count();
        {
            std::lock_guard<std::mutex> lock(g_DirectorMutex);
            auto it = g_InnLastSpoke.find(guid);
            if (it != g_InnLastSpoke.end() && now - it->second < time_t(g_DirectorInnCooldownMinutes) * 60)
                return;
        }

        auto lines = Lines();
        if (!lines)
            return;
        auto it = lines->inn.find(Key(guid, player->GetZoneId()));
        if (it == lines->inn.end())
            return;

        Creature* innkeeper = nullptr;
        NearestInnkeeperCheck check(*player, float(g_DirectorInnRange));
        Acore::CreatureLastSearcher<NearestInnkeeperCheck> searcher(player, innkeeper, check);
        Cell::VisitObjects(player, searcher, float(g_DirectorInnRange));
        if (!innkeeper || innkeeper->IsInCombat() || innkeeper->IsHostileTo(player) || !player->IsWithinLOSInMap(innkeeper))
            return;

        DirectorLine const* line = nullptr;
        {
            std::lock_guard<std::mutex> lock(g_DirectorMutex);
            for (DirectorLine const& candidate : it->second)
                if (!g_InnSaid.count(candidate.id))
                {
                    line = &candidate;
                    break;
                }
            if (!line)
                return;
            g_InnSaid.insert(line->id);
            g_InnLastSpoke[guid] = now;
        }

        // Turned to the one it is talking to, and in Universal for the same reason as a boss: the line was
        // written for this player, and a goblin's greeting that reads as gibberish is not a scene.
        innkeeper->SetFacingToObject(player);
        innkeeper->Say(line->words, LANG_UNIVERSAL, player);

        if (LedgerRecordScene)
            LedgerRecordScene(player, innkeeper, "inn_gossip", line->id);

        OllamaEvents_SceneSpoken(player, innkeeper->GetName(), line->words,
                                 "{0}, the innkeeper, said to {2}: \"{1}\"", g_DirectorInnEventType, false);

        if (g_DebugEnabled)
            LOG_INFO("server.loading", "[Ollama Chat] Director: {} spoke to {} (zone {}, line {})",
                innkeeper->GetName(), player->GetName(), player->GetZoneId(), line->id);
    }
}

OllamaDirectorWorldScript::OllamaDirectorWorldScript()
    : WorldScript("OllamaDirectorWorldScript", { WORLDHOOK_ON_UPDATE })
{
}

void OllamaDirectorWorldScript::OnUpdate(uint32 diff)
{
    static uint32 timer = 0;    // 0: load on the first tick after enabling

    if (!g_DirectorEnable || !(g_DirectorBossScene || g_DirectorInnScene))
        return;

    if (timer > diff)
    {
        timer -= diff;
        return;
    }

    timer = std::max<uint32>(5, g_DirectorRefreshSeconds) * 1000;

    if (World::IsStopped() || g_DirectorLoading.exchange(true))
        return;

    std::thread([]
    {
        LoadLines();
        g_DirectorLoading = false;
    }).detach();
}

OllamaDirectorPlayerScript::OllamaDirectorPlayerScript()
    : PlayerScript("OllamaDirectorPlayerScript", { PLAYERHOOK_ON_UPDATE, PLAYERHOOK_ON_LOGOUT })
{
}

void OllamaDirectorPlayerScript::OnPlayerUpdate(Player* player, uint32 diff)
{
    if (!g_DirectorEnable || !(g_DirectorBossScene || g_DirectorInnScene) || !player || !player->IsInWorld())
        return;

    // Cheap tests first: this runs for every player, bots included, on every tick. Bosses are in dungeons,
    // innkeepers in the open world (battlegrounds and arenas are neither).
    Map* map = player->FindMap();
    if (!map)
        return;
    bool const dungeon = map->IsDungeon();
    if (dungeon ? !g_DirectorBossScene : (!g_DirectorInnScene || map->Instanceable()))
        return;
    if (!OllamaIsRealPlayer(player))
        return;

    {
        std::lock_guard<std::mutex> lock(g_DirectorMutex);
        uint32& elapsed = g_CheckTimers[player->GetGUID().GetCounter()];
        elapsed += diff;
        if (elapsed < 1000)
            return;
        elapsed = 0;
    }

    if (dungeon)
        CheckApproach(player);
    else
        CheckInn(player);
}

void OllamaDirectorPlayerScript::OnPlayerLogout(Player* player)
{
    if (!player)
        return;

    std::lock_guard<std::mutex> lock(g_DirectorMutex);
    g_CheckTimers.erase(player->GetGUID().GetCounter());
    g_InnLastSpoke.erase(player->GetGUID().GetCounter());
}

OllamaDirectorMapScript::OllamaDirectorMapScript()
    : AllMapScript("OllamaDirectorMapScript", { ALLMAPHOOK_ON_DESTROY_INSTANCE })
{
}

void OllamaDirectorMapScript::OnDestroyInstance(MapInstanced* /*mapInstanced*/, Map* map)
{
    if (!map)
        return;

    uint32 const instanceId = map->GetInstanceId();
    std::lock_guard<std::mutex> lock(g_DirectorMutex);
    for (auto it = g_Spoken.begin(); it != g_Spoken.end();)
        it = (uint32(*it >> 32) == instanceId) ? g_Spoken.erase(it) : std::next(it);
}
