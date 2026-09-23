#include "mod-ollama-chat_events.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat_world.h"
#include "mod-ollama-chat_expression.h"
#include "mod-ollama-chat_dispatch.h"
#include "mod-ollama-chat_governor.h"
#include "mod-ollama-chat_handler.h"
#include "mod-ollama-chat_memory.h"
#include "mod-ollama-chat_personality.h"
#include "mod-ollama-chat_roleplay.h"
#include "mod-ollama-chat_sentiment.h"
#include "mod-ollama-chat_response.h"
#include "mod-ollama-chat_topics.h"
#include "mod-ollama-chat-utilities.h"

#include "AchievementMgr.h"
#include "Containers.h"
#include "GameObject.h"
#include "Group.h"
#include "Guild.h"
#include "Item.h"
#include "Log.h"
#include "Map.h"
#include "ObjectAccessor.h"
#include "Player.h"
#include "QuestDef.h"
#include "SpellInfo.h"
#include "SpellMgr.h"

#include "AiFactory.h"
#include "PlayerbotAI.h"
#include "PlayerbotMgr.h"

#include <fmt/core.h>
#include <string>
#include <vector>

namespace
{
    OllamaBotEventChatter eventChatter;

    bool IsGuildEventType(const std::string& type)
    {
        return  type == g_GuildEventTypeGuildJoin      ||
                type == g_GuildEventTypeGuildLeave     ||
                type == g_GuildEventTypeGuildPromotion ||
                type == g_GuildEventTypeGuildDemotion  ||
                type == g_GuildEventTypeLevelUp        ||
                type == g_GuildEventTypeEpicGear       ||
                type == g_GuildEventTypeRareGear       ||
                type == g_GuildEventTypeDungeonComplete||
                type == g_GuildEventTypeGuildAchievement ||
                type == g_GuildEventTypeGuildLogin;
    }

    bool GuildHasRealPlayerOnline(uint32 guildId)
    {
        if (!guildId)
            return false;

        for (auto const& pair : ObjectAccessor::GetPlayers())
        {
            Player* player = pair.second;
            if (!player || !player->IsInWorld())
                continue;
            if (OllamaIsBotPlayer(player))
                continue;
            if (player->GetGuildId() == guildId)
                return true;
        }
        return false;
    }

    int EventChanceFor(const std::string& type)
    {
        // Ordinary events.
        if (type == g_EventTypeLearnedSpell)    return g_EventTypeLearnedSpell_Chance;
        if (type == g_EventTypeDefeatedBoss)    return g_EventTypeDefeatedBoss_Chance;
        if (type == g_EventTypeDefeated)        return g_EventTypeDefeated_Chance;
        if (type == g_EventTypeDefeatedPlayer)  return g_EventTypeDefeatedPlayer_Chance;
        if (type == g_EventTypePetDefeated)     return g_EventTypePetDefeated_Chance;
        if (type == g_EventTypeGotItem)         return g_EventTypeGotItem_Chance;
        if (type == g_EventTypeDied)            return g_EventTypeDied_Chance;
        if (type == g_EventTypeCompletedQuest)  return g_EventTypeCompletedQuest_Chance;
        if (type == g_EventTypeRequestedDuel)   return g_EventTypeRequestedDuel_Chance;
        if (type == g_EventTypeStartedDueling)  return g_EventTypeStartedDueling_Chance;
        if (type == g_EventTypeWonDuel)         return g_EventTypeWonDuel_Chance;
        if (type == g_EventTypeLeveledUp)       return g_EventTypeLeveledUp_Chance;
        if (type == g_EventTypeAchievement)     return g_EventTypeAchievement_Chance;
        if (type == g_EventTypeUsedObject)      return g_EventTypeUsedObject_Chance;

        // Guild events.
        if (type == g_GuildEventTypeEpicGear)         return g_GuildEventTypeEpicGear_Chance;
        if (type == g_GuildEventTypeRareGear)         return g_GuildEventTypeRareGear_Chance;
        if (type == g_GuildEventTypeGuildJoin)        return g_GuildEventTypeGuildJoin_Chance;
        if (type == g_GuildEventTypeGuildLogin)       return g_GuildEventTypeGuildLogin_Chance;
        if (type == g_GuildEventTypeGuildLeave)       return g_GuildEventTypeGuildLeave_Chance;
        if (type == g_GuildEventTypeGuildPromotion)   return g_GuildEventTypeGuildPromotion_Chance;
        if (type == g_GuildEventTypeGuildDemotion)    return g_GuildEventTypeGuildDemotion_Chance;
        if (type == g_GuildEventTypeGuildAchievement) return g_GuildEventTypeGuildAchievement_Chance;
        if (type == g_GuildEventTypeLevelUp)          return g_GuildEventTypeLevelUp_Chance;
        if (type == g_GuildEventTypeDungeonComplete)  return g_GuildEventTypeDungeonComplete_Chance;

        return 0;
    }

    // Plain-language line for the witnessed-event memory that the topic engine
    // reads. This is what lets a bot say "that hawk nearly had you" instead of
    // reciting its own spell list.
    std::string MemoryLineFor(const std::string& actor, const std::string& type,
                              const std::string& detail)
    {
        if (type == g_EventTypeDefeatedBoss)
            return SafeFormat("{} brought down {}, the master of this place", actor, detail);
        if (type == g_EventTypeDefeated || type == g_EventTypePetDefeated)
            return SafeFormat("{} killed {}", actor, detail);
        if (type == g_EventTypeDefeatedPlayer)
            return SafeFormat("{} cut down {} in a fight", actor, detail);
        if (type == g_EventTypeDied)
            return detail.empty() ? SafeFormat("{} was killed", actor)
                                  : SafeFormat("{} was killed by {}", actor, detail);
        if (type == g_EventTypeGotItem)
            return SafeFormat("{} picked up {}", actor, detail);
        if (type == g_EventTypeCompletedQuest)
            return SafeFormat("{} finished the task '{}'", actor, detail);
        if (type == g_EventTypeLeveledUp)
            return detail.empty() ? SafeFormat("{} grew stronger", actor)
                                  : SafeFormat("{} grew stronger, now level {}", actor, detail);
        if (type == g_EventTypeAchievement)
            return SafeFormat("{} earned recognition for {}", actor, detail);
        if (type == g_EventTypeWonDuel)
            return SafeFormat("{} beat {} in a duel", actor, detail);
        if (type == g_EventTypeLearnedSpell)
            return SafeFormat("{} learned {}", actor, detail);
        return "";
    }

    // Which deeds are worth remembering (plan 38).
    //
    // Deliberately NOT g_EventTypeDefeated: ordinary kills are 48,000 a day on
    // this realm and a dungeon's elites are its trash, so a rank alone is no
    // signal. A boss, a death, a task finished, a prize taken, a person cut
    // down -- those are the things anyone would still be telling afterwards.
    //
    // And deliberately NOT g_EventTypeAchievement, as of plans/44 §5. An achievement is not a deed; it is
    // a notice ABOUT deeds, and a dungeon-clear one fires in the same second as the final boss dies, once
    // per party member. Five near-identical "X earned recognition for Gnomeregan" lines then crowd the
    // single boss line out of a digest that writes three to five notes -- which is why Thermaplugg, Bazil
    // Thredd and Edwin VanCleef are the only bosses this realm has ever failed to remember, and why the
    // notes that replaced them invented a story ("<the player> joined the ranks of the mad"). The dispatch is
    // gone too (see ChatOnAchievement below); this is the second half of the same fix.
    bool IsMemorableEvent(const std::string& type)
    {
        return type == g_EventTypeDefeatedBoss   || type == g_EventTypeDied ||
               type == g_EventTypeCompletedQuest || type == g_EventTypeDefeatedPlayer ||
               type == g_EventTypeGotItem        || type == g_EventTypeLeveledUp ||
               type == g_EventTypeWonDuel;
    }

    // Where this happened, told the way a person would tell it: the dungeon's
    // own name when inside one, the zone otherwise. The area accessors live on
    // PlayerbotAI, so this only answers for bots -- which is all it is asked.
    std::string PlaceNameFor(Player* bot)
    {
        if (!bot)
            return "";

        if (Map* map = bot->GetMap(); map && map->IsDungeon())
            return map->GetMapName();

        PlayerbotAI* ai = PlayerbotsMgr::instance().GetPlayerbotAI(bot);
        if (!ai)
            return "";

        AreaTableEntry const* zone = ai->GetCurrentZone();
        return zone ? PlayerbotAI::GetLocalizedAreaName(zone) : "";
    }

    // Hand the deed to every bot who was there to see it, the actor included --
    // a bot that dies four times in a marsh should remember it above all.
    void BroadcastEventMemory(Player* actor, const std::string& line, float radius)
    {
        if (!actor || line.empty() || !g_MemoryEnable || !g_MemoryEventEnable)
            return;

        auto note = [&line](Player* witness)
        {
            PlayerbotAI* ai = PlayerbotsMgr::instance().GetPlayerbotAI(witness);
            if (!ai || !ai->IsBotAI())
                return;

            // Standing in a company with a person makes this one a companion from here on -- including
            // for the nights it spends out alone, which is what makes it worth asking where it has been.
            if (OllamaGroupHasRealPlayer(witness))
                Memory_NoteCompanion(witness->GetGUID().GetRawValue());

            const std::string place = PlaceNameFor(witness);
            Memory_NoteGameEvent(witness->GetGUID().GetRawValue(),
                                 place.empty() ? line : line + ", in " + place);
        };

        note(actor);

        for (auto const& pair : ObjectAccessor::GetPlayers())
        {
            Player* witness = pair.second;
            if (!witness || witness == actor || !witness->IsInWorld())
                continue;
            if (witness->GetMap() != actor->GetMap())
                continue;
            if (!actor->IsWithinDistInMap(witness, radius))
                continue;

            note(witness);
        }
    }
}

// --------------------------------------------------------------------------

void OllamaBotEventChatter::DispatchGameEvent(Player* source, std::string type, std::string detail)
{
    if (!g_Enable || !g_EnableEventChatter || !source || type.empty())
        return;

    if (!source->IsInWorld() || !source->GetMap())
        return;

    const bool sourceIsBot = OllamaIsBotPlayer(source);

    // Seed the witnessed-event memory before any chance roll: bots should
    // remember what they saw even when they choose not to comment on it.
    if (const std::string memory = MemoryLineFor(source->GetName(), type, detail); !memory.empty())
    {
        Topics_BroadcastEventToNearby(source, memory, g_EventChatterRealPlayerDistance);

        // The topic engine keeps this only long enough to talk about. A deed
        // worth remembering also goes to the memory store, which outlives the
        // conversation -- and unlike condensation, it does not need anyone to
        // have said a word about it (plan 38).
        if (IsMemorableEvent(type))
            BroadcastEventMemory(source, memory, g_EventChatterRealPlayerDistance);
    }

    const bool isGuildEvent = source->GetGuild() && g_EnableGuildEventChatter &&
                              IsGuildEventType(type) &&
                              GuildHasRealPlayerOnline(source->GetGuildId());

    bool hasNearbyRealPlayer = false;
    for (auto const& pair : source->GetMap()->GetPlayers())
    {
        Player* player = pair.GetSource();
        if (!player || player == source)
            continue;
        if (OllamaIsBotPlayer(player))
            continue;
        if (player->IsWithinDist(source, g_EventChatterRealPlayerDistance, false))
        {
            hasNearbyRealPlayer = true;
            break;
        }
    }

    if (sourceIsBot && !hasNearbyRealPlayer && !isGuildEvent)
        return;

    // The chance roll happens BEFORE any cooldown is consumed. The old code
    // stamped cooldowns during candidate filtering and then early-returned
    // here, so bots burned their event cooldown on events that were discarded.
    const int chance = EventChanceFor(type);
    if (chance <= 0)
        return;
    if (int(urand(1, 100)) > chance)
        return;

    if (g_DebugEnabled)
        LOG_INFO("module.ollamachat", "[Ollama Chat] Event from {}: type={} detail={}",
                 source->GetName(), type, detail);

    // Gather candidates.
    std::vector<Player*> candidateBots;

    if (isGuildEvent)
    {
        const uint32 guildId = source->GetGuildId();
        for (auto const& pair : ObjectAccessor::GetPlayers())
        {
            Player* player = pair.second;
            if (!player || !player->IsInWorld() || !player->IsAlive() ||
                player->GetGuildId() != guildId)
                continue;

            PlayerbotAI* ai = PlayerbotsMgr::instance().GetPlayerbotAI(player);
            if (ai && ai->IsBotAI())
                candidateBots.push_back(player);
        }
    }
    else
    {
        for (auto const& pair : source->GetMap()->GetPlayers())
        {
            Player* player = pair.GetSource();
            if (!player || !player->IsWithinDist(source, g_EventChatterRealPlayerDistance, false))
                continue;

            // A corpse is not a candidate. This also removes the dying bot's
            // own comment on its death -- it is dead by the time OnUnitDeath
            // runs -- which is the right voice to lose: the ones who should
            // be talking are the ones still standing.
            if (!player->IsAlive())
                continue;

            PlayerbotAI* ai = PlayerbotsMgr::instance().GetPlayerbotAI(player);
            if (ai && ai->IsBotAI())
                candidateBots.push_back(player);
        }
    }

    if (candidateBots.empty())
        return;

    Acore::Containers::RandomShuffle(candidateBots);

    const uint32_t maxBots = isGuildEvent ? g_GuildChatterMaxBotsPerEvent
                                          : g_EventChatterMaxBotsPerPlayer;
    uint32_t responses = 0;

    for (Player* bot : candidateBots)
    {
        // Events are reactions to something that just happened, and in a fight that is when the most
        // worth reacting to happens. This used to be gated with everything else.
        if (!g_CombatEvents && bot->IsInCombat())
            continue;

        uint32_t botChance;
        if (bot == source)
            botChance = g_EventChatterBotSelfCommentChance;
        else if (isGuildEvent)
            botChance = g_GuildChatterBotCommentChance;
        else
            botChance = g_EventChatterBotCommentChance;

        if (botChance == 0 || urand(1, 100) > botChance)
            continue;

        // Cooldown is now keyed by ObjectGuid, pruned on logout, and consumed
        // only here -- at the point the bot actually commits to speaking.
        if (!Governor_TryConsumeEventCooldown(bot->GetGUID()))
            continue;

        // Was: any group at all, which let a bot spend a generation
        // commenting to a party of nothing but bots. Say is the fallback, and
        // the loop above already established a real player is in range of it.
        const bool partyAudience =
            bot->GetGroup() && !g_DisableForParty && OllamaGroupHasRealPlayer(bot);

        const ChatChannelSourceLocal source_ =
            isGuildEvent ? SRC_GUILD_LOCAL
                         : (partyAudience ? SRC_PARTY_LOCAL : SRC_SAY_LOCAL);

        // Party events key on the GROUP, exactly as a party line does in
        // ProcessChat. Keying them on the zone put a bot's remark about your
        // loot in a different conversation space from the party chat it was
        // said in -- so it counted for no cooldown, no repetition history and,
        // since plan 25 item 54, no thread: a bot could remark on your drop and
        // still not be who you were talking to.
        uint32_t scopeGroupOrZone = bot->GetZoneId();
        if (source_ == SRC_PARTY_LOCAL)
        {
            if (Group* botGroup = bot->GetGroup())
                scopeGroupOrZone = botGroup->GetGUID().GetCounter();
        }

        const std::string scopeKey = Governor_MakeScopeKey(
            ChatChannelSourceLocalStr[source_], 0, "",
            source_ == SRC_GUILD_LOCAL ? bot->GetGuildId() : 0,
            scopeGroupOrZone);

        if (!Governor_CanSend(bot->GetGUID(), scopeKey))
            continue;

        // Built here, on the world thread. The old code built the whole prompt
        // inside the worker, reading area, zone, spec and guild off-thread.
        uint32_t maxWords = 0;
        std::string prompt = BuildPrompt(bot, g_EventChatterPromptTemplate, type, detail,
                                         source->GetName(), &maxWords);
        if (prompt.empty())
            continue;

        OllamaChatRequest request;
        request.botGuid     = bot->GetGUID().GetRawValue();
        request.targetGuid  = (bot == source) ? 0 : source->GetGUID().GetRawValue();
        request.source      = source_;
        request.chainDepth  = 0;
        request.scopeKey    = scopeKey;
        request.prompt      = std::move(prompt);
        request.botName     = bot->GetName();
        request.kind        = OllamaRequestKind::EventChatter;
        request.maxWords    = maxWords;
        request.triggerBotReplies = true;

        if (!OllamaDispatch_Submit(std::move(request)))
            continue;

        ++responses;
        if (maxBots > 0 && responses >= maxBots)
            break;
    }

    if (g_DebugEnabled)
        LOG_INFO("module.ollamachat", "[Ollama Chat] Event dispatch complete, {} bots queued.", responses);
}

std::string OllamaBotEventChatter::BuildPrompt(Player* bot, std::string promptTemplate,
                                               std::string eventType, std::string eventDetail,
                                               std::string actorName, uint32_t* outMaxWords)
{
    if (!bot || promptTemplate.empty())
        return "";

    PlayerbotAI* ai = PlayerbotsMgr::instance().GetPlayerbotAI(bot);
    if (!ai || !ai->GetChatHelper())
        return "";

    const std::string personality       = GetBotPersonality(bot);
    const std::string personalityPrompt = GetPersonalityPromptAddition(personality);

    AreaTableEntry const* area = ai->GetCurrentArea();
    AreaTableEntry const* zone = ai->GetCurrentZone();

    Player* actor = actorName.empty() ? nullptr : ObjectAccessor::FindPlayerByName(actorName);
    std::string sentimentInfo;
    if ((g_RegardEnable || g_EnableSentimentTracking) && actor)
        sentimentInfo = g_RegardEnable ? Regard_WordsFor(bot, actor) : GetSentimentPromptAddition(bot, actor);

    std::string prompt = SafeFormat(
        promptTemplate,
        fmt::arg("bot_name", bot->GetName()),
        fmt::arg("bot_level", bot->GetLevel()),
        fmt::arg("bot_class", ai->GetChatHelper()->FormatClass(bot->getClass())),
        fmt::arg("bot_race", ai->GetChatHelper()->FormatRace(bot->getRace())),
        fmt::arg("bot_gender", bot->getGender() == GENDER_MALE ? "Male" : "Female"),
        fmt::arg("bot_role", CleanRoleForPrompt(ChatHelper::FormatClass(bot, AiFactory::GetPlayerSpecTab(bot)))),
        fmt::arg("bot_faction", bot->GetTeamId() == TEAM_ALLIANCE ? "Alliance" : "Horde"),
        fmt::arg("bot_area", area ? PlayerbotAI::GetLocalizedAreaName(area) : "UnknownArea"),
        fmt::arg("bot_zone", zone ? PlayerbotAI::GetLocalizedAreaName(zone) : "UnknownZone"),
        fmt::arg("bot_map", OllamaContinentName(bot)),
        fmt::arg("bot_personality", personalityPrompt),
        fmt::arg("bot_personality_name", personality),
        fmt::arg("event_type", eventType),
        fmt::arg("event_detail", eventDetail),
        fmt::arg("actor_name", actorName),
        fmt::arg("sentiment_info", sentimentInfo));

    prompt += Memory_BuildPromptSection(bot, nullptr);
    prompt += Regard_PromptSection(bot, nullptr);
    prompt += Regard_CompanySection(bot, actor, false);
    prompt += Roleplay_BuildVoicePrompt(bot);
    prompt += Expression_BuildGesturePrompt();

    // Drawn last, like the reply and emote paths, so it is the final
    // instruction the model reads. The "@N" comes back out so ClampReplyWords
    // can hold the reaction to it; the template's own "under 20 words" has been
    // removed, because a fixed length there beats whatever is drawn here.
    if (!g_EventRegisters.empty())
    {
        std::string reg = g_EventRegisters[urand(0, static_cast<uint32_t>(g_EventRegisters.size() - 1))];
        const uint32_t cap = TakeWordCap(reg);
        if (outMaxWords)
            *outMaxWords = cap;
        prompt += " " + reg;
    }

    return prompt;
}

// ==========================================================================
// Script hooks
// ==========================================================================

ChatOnKill::ChatOnKill()
    : PlayerScript("ChatOnKill", {
          PLAYERHOOK_ON_CREATURE_KILL,
          PLAYERHOOK_ON_PVP_KILL,
          PLAYERHOOK_ON_CREATURE_KILLED_BY_PET,
      }) { }

void ChatOnKill::OnPlayerCreatureKill(Player* killer, Creature* victim)
{
    if (!killer || !victim)
        return;

    // A boss going down is the largest thing that happens in a night's play, and it used to be dispatched
    // as the same event as a boar: EventTypeDefeated at a 1% chance against 494,622 logged kills, so the
    // party fell silent at the one moment worth speaking (plans/30 §3). Its own event, its own chance.
    // rank 3 is a world boss, the same reading mod-ledger takes of creature_template.rank.
    CreatureTemplate const* proto = victim->GetCreatureTemplate();
    const bool master = victim->IsDungeonBoss() || (proto && proto->rank == 3);

    // A floor under the ordinary kill (plan 25 item 33). Nobody remarks on stepping on a chicken, and
    // nobody remarks on a wolf that a veteran killed without breaking stride. Bosses are exempt: a master
    // of the place is worth saying regardless of how easily it fell. This gate sits ahead of
    // DispatchGameEvent deliberately, because the witnessed-event memory is seeded there BEFORE any
    // chance roll -- so flooring only the roll would still broadcast "X slew Chicken" to every bot nearby.
    if (!master)
    {
        if (victim->IsCritter())
            return;

        const int gap = int(killer->GetLevel()) - int(victim->GetLevel());
        if (g_EventDefeatedTrivialLevelGap > 0 && gap >= g_EventDefeatedTrivialLevelGap)
            return;
    }

    eventChatter.DispatchGameEvent(killer, master ? g_EventTypeDefeatedBoss : g_EventTypeDefeated,
                                   victim->GetName());
}

void ChatOnKill::OnPlayerPVPKill(Player* killer, Player* killed)
{
    if (killer && killed)
        eventChatter.DispatchGameEvent(killer, g_EventTypeDefeatedPlayer, killed->GetName());
}

void ChatOnKill::OnPlayerCreatureKilledByPet(Player* owner, Creature* victim)
{
    if (owner && victim)
        eventChatter.DispatchGameEvent(owner, g_EventTypePetDefeated, victim->GetName());
}

// --------------------------------------------------------------------------

ChatOnLoot::ChatOnLoot()
    : PlayerScript("ChatOnLoot", { PLAYERHOOK_ON_STORE_NEW_ITEM }) { }

void ChatOnLoot::OnPlayerStoreNewItem(Player* player, Item* item, uint32 /*count*/)
{
    if (!player || !item || !item->GetTemplate())
        return;

    ItemTemplate const* tmpl = item->GetTemplate();

    // Blue or better (plans/44 §11). This was UNCOMMON, and because the memory is seeded above the chance
    // roll, EventTypeGotItem_Chance never governed it: every green a party picked up became a deed in every
    // nearby bot's buffer. Measured on the Gnomeregan night: 300 of 890 buffered deeds were prizes, and 85
    // of the party's 199 memories were loot announcements -- 62 of them about what the player looted. That
    // is the mass a boss kill has to compete with inside one digest.
    if (tmpl->Quality >= ITEM_QUALITY_RARE)
        eventChatter.DispatchGameEvent(player, g_EventTypeGotItem, tmpl->Name1);

    if (!player->GetGuild() || !g_EnableGuildEventChatter)
        return;

    if (tmpl->Quality == ITEM_QUALITY_EPIC && !g_GuildEventTypeEpicGear.empty())
    {
        eventChatter.DispatchGameEvent(player, g_GuildEventTypeEpicGear, tmpl->Name1);
    }
    else if (tmpl->Quality == ITEM_QUALITY_RARE && !g_GuildEventTypeRareGear.empty())
    {
        if (tmpl->Class == ITEM_CLASS_WEAPON || tmpl->Class == ITEM_CLASS_ARMOR)
            eventChatter.DispatchGameEvent(player, g_GuildEventTypeRareGear, tmpl->Name1);
    }
}

// --------------------------------------------------------------------------

ChatOnDeath::ChatOnDeath()
    : UnitScript("ChatOnDeath", true, { UNITHOOK_ON_UNIT_DEATH }) { }

void ChatOnDeath::OnUnitDeath(Unit* unit, Unit* killer)
{
    if (!unit || !unit->IsPlayer())
        return;

    Player* victim = unit->ToPlayer();
    Player* killerPlayer = killer ? killer->GetCharmerOrOwnerPlayerOrPlayerItself() : nullptr;
    if (killerPlayer == victim)
        killerPlayer = nullptr;

    // Who or what took them. Left empty for a fall, deep water or the cold:
    // there is no name to give, and the memory line reads "was killed" as it
    // always did. Anything else names the killer, so a witness can say what it
    // actually saw instead of "someone died".
    std::string detail;
    if (killer && killer != victim)
        detail = killerPlayer ? killerPlayer->GetName() : killer->GetName();

    eventChatter.DispatchGameEvent(victim, g_EventTypeDied, detail);
}

// --------------------------------------------------------------------------

ChatOnQuest::ChatOnQuest()
    : PlayerScript("ChatOnQuest", { PLAYERHOOK_ON_PLAYER_COMPLETE_QUEST }) { }

void ChatOnQuest::OnPlayerCompleteQuest(Player* player, Quest const* quest)
{
    if (!player || !quest)
        return;

    eventChatter.DispatchGameEvent(player, g_EventTypeCompletedQuest, quest->GetTitle());

    if (player->GetGuild() && g_EnableGuildEventChatter &&
        !g_GuildEventTypeDungeonComplete.empty() &&
        player->GetMap() && player->GetMap()->IsDungeon())
    {
        eventChatter.DispatchGameEvent(
            player, g_GuildEventTypeDungeonComplete,
            SafeFormat("{} in {}", quest->GetTitle(), OllamaContinentName(player)));
    }
}

// --------------------------------------------------------------------------

ChatOnLearn::ChatOnLearn()
    : PlayerScript("ChatOnLearn", { PLAYERHOOK_ON_LEARN_SPELL }) { }

void ChatOnLearn::OnPlayerLearnSpell(Player* player, uint32 spellID)
{
    if (!player)
        return;

    SpellInfo const* spellInfo = sSpellMgr->GetSpellInfo(spellID);
    if (spellInfo && spellInfo->SpellName[0] && *spellInfo->SpellName[0])
        eventChatter.DispatchGameEvent(player, g_EventTypeLearnedSpell, spellInfo->SpellName[0]);
}

// --------------------------------------------------------------------------

ChatOnDuel::ChatOnDuel()
    : PlayerScript("ChatOnDuel", {
          PLAYERHOOK_ON_DUEL_REQUEST,
          PLAYERHOOK_ON_DUEL_START,
          PLAYERHOOK_ON_DUEL_END,
      }) { }

void ChatOnDuel::OnPlayerDuelRequest(Player* target, Player* challenger)
{
    if (challenger && target)
        eventChatter.DispatchGameEvent(challenger, g_EventTypeRequestedDuel, target->GetName());
}

void ChatOnDuel::OnPlayerDuelStart(Player* player1, Player* player2)
{
    if (player1 && player2)
        eventChatter.DispatchGameEvent(player1, g_EventTypeStartedDueling, player2->GetName());
}

void ChatOnDuel::OnPlayerDuelEnd(Player* winner, Player* loser, DuelCompleteType /*type*/)
{
    if (winner && loser)
        eventChatter.DispatchGameEvent(winner, g_EventTypeWonDuel, loser->GetName());
}

// --------------------------------------------------------------------------

ChatOnLevelUp::ChatOnLevelUp()
    : PlayerScript("ChatOnLevelUp", { PLAYERHOOK_ON_LEVEL_CHANGED }) { }

void ChatOnLevelUp::OnPlayerLevelChanged(Player* player, uint8 /*oldLevel*/)
{
    if (!player)
        return;

    // Local patch (custom-wow): at hard roleplay the level is not something a
    // person in the world could name, so the event carries no figure.
    const std::string level = (g_RoleplayEnable && g_RoleplayStrictness >= 2)
                                  ? std::string()
                                  : std::to_string(player->GetLevel());
    eventChatter.DispatchGameEvent(player, g_EventTypeLeveledUp, level);

    if (player->GetGuild() && g_EnableGuildEventChatter && !g_GuildEventTypeLevelUp.empty())
        eventChatter.DispatchGameEvent(player, g_GuildEventTypeLevelUp, level);
}

// --------------------------------------------------------------------------

ChatOnAchievement::ChatOnAchievement()
    : PlayerScript("ChatOnAchievement", { PLAYERHOOK_ON_ACHI_COMPLETE }) { }

void ChatOnAchievement::OnPlayerAchievementComplete(Player* /*player*/, AchievementEntry const* /*achievement*/)
{
    // Achievements are no longer hooked into the chat system at all -- the operator's ruling, plans/44 §11.
    //
    // Setting EventTypeAchievement_Chance to 0 would NOT have done this: the witnessed-event memory is
    // seeded above the chance roll (DispatchGameEvent, deliberately), so a silenced achievement still
    // filled every nearby bot's deed buffer. The collision it caused is written up in plans/44 §5.
    //
    // An achievement is also the one event type with no in-world referent: "earned recognition for
    // Gnomeregan" is the game talking about itself, which is exactly what the system prompt tells these
    // characters they know nothing of.
    //
    // The hook stays registered and empty on purpose, so that anyone restoring this can see what was here.
    // Two lines bring it back (the personal and guild dispatches), and the guild announcement went with
    // them -- it was not the cause, but "at all" was the instruction.
}

// --------------------------------------------------------------------------

ChatOnGameObjectUse::ChatOnGameObjectUse() : AllGameObjectScript("ChatOnGameObjectUse") { }

bool ChatOnGameObjectUse::CanGameObjectGossipHello(Player* player, GameObject* go)
{
    if (player && go && go->GetGOInfo())
        eventChatter.DispatchGameEvent(player, g_EventTypeUsedObject, go->GetGOInfo()->name);

    // false = we did not handle the interaction; normal processing continues.
    return false;
}

// --------------------------------------------------------------------------

ChatOnGuild::ChatOnGuild()
    : GuildScript("ChatOnGuild", {
          GUILDHOOK_ON_ADD_MEMBER,
          GUILDHOOK_ON_REMOVE_MEMBER,
          GUILDHOOK_ON_EVENT,
      }) { }

void ChatOnGuild::OnAddMember(Guild* guild, Player* player, uint8& /*plRank*/)
{
    if (!guild || !player || !g_EnableGuildEventChatter || g_GuildEventTypeGuildJoin.empty())
        return;

    eventChatter.DispatchGameEvent(player, g_GuildEventTypeGuildJoin, guild->GetName());
}

void ChatOnGuild::OnRemoveMember(Guild* guild, Player* player, bool /*isDisbanding*/, bool /*isKicked*/)
{
    if (!guild || !player || !g_EnableGuildEventChatter || g_GuildEventTypeGuildLeave.empty())
        return;

    eventChatter.DispatchGameEvent(player, g_GuildEventTypeGuildLeave, guild->GetName());
}

void ChatOnGuild::OnEvent(Guild* guild, uint8 eventType, ObjectGuid::LowType playerGuid1,
                          ObjectGuid::LowType /*playerGuid2*/, uint8 newRank)
{
    if (!guild || !g_EnableGuildEventChatter)
        return;

    if (eventType != GUILD_EVENT_LOG_PROMOTE_PLAYER && eventType != GUILD_EVENT_LOG_DEMOTE_PLAYER)
        return;

    Player* player = ObjectAccessor::FindConnectedPlayer(
        ObjectGuid::Create<HighGuid::Player>(playerGuid1));
    if (!player)
        return;

    const bool promoted = (eventType == GUILD_EVENT_LOG_PROMOTE_PLAYER);
    const std::string& type = promoted ? g_GuildEventTypeGuildPromotion
                                       : g_GuildEventTypeGuildDemotion;
    if (type.empty())
        return;

    eventChatter.DispatchGameEvent(player, type, std::to_string(newRank));
}

// --------------------------------------------------------------------------

ChatOnGuildLogin::ChatOnGuildLogin()
    : PlayerScript("ChatOnGuildLogin", { PLAYERHOOK_ON_LOGIN }) { }

void ChatOnGuildLogin::OnPlayerLogin(Player* player)
{
    if (!player || !g_EnableGuildEventChatter || g_GuildEventTypeGuildLogin.empty())
        return;

    Guild* guild = player->GetGuild();
    if (!guild)
        return;

    // Only real players; a wave of bot logins would spam the guild channel.
    // Must be the session test: the bot's AI is not attached yet at this point.
    if (OllamaIsBotPlayer(player))
        return;

    eventChatter.DispatchGameEvent(player, g_GuildEventTypeGuildLogin, guild->GetName());
}
