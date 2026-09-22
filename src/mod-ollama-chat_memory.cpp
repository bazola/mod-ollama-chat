#include "mod-ollama-chat_memory.h"
#include "mod-ollama-chat_api.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat_dispatch.h"
#include "mod-ollama-chat_handler.h"
#include "mod-ollama-chat_personality.h"
#include "mod-ollama-chat_roleplay.h"
#include "mod-ollama-chat-utilities.h"

#include "Config.h"
#include "DatabaseEnv.h"
#include "Log.h"
#include "ObjectAccessor.h"
#include "Player.h"

#include "PlayerbotAI.h"
#include "PlayerbotMgr.h"

#include <fmt/core.h>

#include <algorithm>
#include <cctype>
#include <ctime>
#include <mutex>
#include <unordered_set>
#include <unordered_map>

namespace
{
    std::mutex g_mutex;

    struct OllamaMemoryState
    {
        std::vector<BotMemoryEntry>                    memories;
        std::unordered_map<uint64_t, BotRelationship>  relationships;
        bool condensing = false;    // a job is already in flight
        std::unordered_map<uint64_t, bool> relationshipPending;
        bool dirty = false;

        // Deeds witnessed since the last digest (plan 38), with the time the
        // first one landed so a part-filled buffer can be flushed when stale.
        std::vector<std::string> eventBuffer;
        uint64_t                 eventFirstAt  = 0;
        bool                     eventFlushing = false;
    };

    std::unordered_map<uint64_t, OllamaMemoryState> g_state;

    // Bots that have travelled with a person, and so keep deed memories (plan 38).
    std::mutex                    g_CompanionMutex;
    std::unordered_set<uint64_t>  g_Companions;

    std::string Escape(std::string v)
    {
        CharacterDatabase.EscapeString(v);
        return v;
    }

    uint64_t NowSeconds()
    {
        return static_cast<uint64_t>(time(nullptr));
    }

    // Count whole-word, case-insensitive occurrences of `name` in `text`.
    uint32_t CountMentions(const std::string& text, const std::string& name)
    {
        if (name.empty() || text.empty())
            return 0;

        auto lower = [](std::string v)
        {
            for (char& c : v)
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return v;
        };

        const std::string hay = lower(text);
        const std::string needle = lower(name);

        uint32_t count = 0;
        size_t pos = 0;
        while ((pos = hay.find(needle, pos)) != std::string::npos)
        {
            const bool startOk = (pos == 0) ||
                                 !std::isalnum(static_cast<unsigned char>(hay[pos - 1]));
            const size_t end = pos + needle.size();
            const bool endOk = (end >= hay.size()) ||
                               !std::isalnum(static_cast<unsigned char>(hay[end]));
            if (startOk && endOk)
                ++count;
            pos = end;
        }
        return count;
    }

    // Total tokens currently held in a bot's raw conversation history.
    uint32_t HistoryTokens(uint64_t botGuid)
    {
        uint32_t total = 0;

        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        auto botIt = g_BotConversationHistory.find(botGuid);
        if (botIt == g_BotConversationHistory.end())
            return 0;

        for (const auto& [playerGuid, history] : botIt->second)
            for (const auto& pair : history)
                total += Memory_EstimateTokens(pair.playerMessage) + Memory_EstimateTokens(pair.botReply);

        return total;
    }

    std::string RenderHistory(uint64_t botGuid)
    {
        std::string out;

        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        auto botIt = g_BotConversationHistory.find(botGuid);
        if (botIt == g_BotConversationHistory.end())
            return out;

        for (const auto& [playerGuid, history] : botIt->second)
        {
            Player* other = ObjectAccessor::FindPlayer(ObjectGuid(playerGuid));
            const std::string name = other ? other->GetName() : "someone";

            for (const auto& pair : history)
            {
                if (!pair.playerMessage.empty())
                    out += name + ": " + pair.playerMessage + "\n";
                if (!pair.botReply.empty())
                    out += "You: " + pair.botReply + "\n";
            }
        }
        return out;
    }

    // After condensing, drop what has been distilled but keep the turns the prompt still shows. Erasing the
    // lot would make a bot go blank about the last thing said to it in the middle of a conversation --
    // exactly the failure the condenser exists to cure, moved from between sessions to inside one.
    void ClearHistory(uint64_t botGuid)
    {
        {
            std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
            auto it = g_BotConversationHistory.find(botGuid);
            if (it != g_BotConversationHistory.end())
            {
                for (auto& [playerGuid, turns] : it->second)
                    while (turns.size() > g_MaxConversationHistory)
                        turns.pop_front();

                // The trigger counts every conversation this bot is holding, not one of them, so keeping a
                // full prompt window for each would leave a busy bot still over the threshold the moment it
                // finished condensing -- and it would condense again on the very next line, once per
                // exchange, for ever. At ~44 tokens a turn that starts at about seven people talking to the
                // same bot at once. Drop the oldest turn from the longest conversation until there is real
                // headroom, so the next condensation needs new words to reach it.
                const uint32_t headroom = g_MemoryHistoryTokenLimit / 2;
                while (g_MemoryHistoryTokenLimit > 0)
                {
                    uint32_t total = 0;
                    std::deque<BotConversationEntry>* longest = nullptr;
                    for (auto& [playerGuid, turns] : it->second)
                    {
                        for (const BotConversationEntry& turn : turns)
                            total += Memory_EstimateTokens(turn.playerMessage)
                                   + Memory_EstimateTokens(turn.botReply);
                        if (!turns.empty() && (!longest || turns.size() > longest->size()))
                            longest = &turns;
                    }
                    if (total < headroom || !longest)
                        break;
                    longest->pop_front();
                }

                // The DELETE below takes every row for this bot, so the turns we keep have to be written
                // again or a restart would lose them.
                for (auto& [playerGuid, turns] : it->second)
                    for (BotConversationEntry& turn : turns)
                        turn.persisted = false;
            }
        }

        // The rows outlived the in-memory window until now, so every restart
        // reloaded a conversation that had already been condensed and paid to
        // condense it all over again.
        DeleteBotConversationHistoryFromDB(botGuid);
    }

    // Parse the model's condensation output. One memory per line, optionally
    // prefixed with an importance score: "7 | He saved my life at the bridge."
    std::vector<BotMemoryEntry> ParseMemories(const std::string& text)
    {
        std::vector<BotMemoryEntry> out;

        size_t start = 0;
        while (start <= text.size())
        {
            size_t end = text.find('\n', start);
            if (end == std::string::npos)
                end = text.size();

            std::string line = text.substr(start, end - start);
            start = end + 1;

            // Trim and strip list markers the model adds unasked.
            size_t a = line.find_first_not_of(" \t\r-*•");
            if (a == std::string::npos)
                continue;
            size_t b = line.find_last_not_of(" \t\r");
            line = line.substr(a, b - a + 1);
            if (line.empty())
                continue;

            BotMemoryEntry entry;
            entry.importance = 5;

            // Leading "N |" or "N."
            size_t sep = line.find('|');
            if (sep != std::string::npos && sep <= 3)
            {
                try
                {
                    const int score = std::stoi(line.substr(0, sep));
                    entry.importance = static_cast<uint8_t>(std::clamp(score, 1, 10));
                    line = line.substr(sep + 1);
                }
                catch (const std::exception&) { }
            }

            a = line.find_first_not_of(" \t");
            if (a == std::string::npos)
                continue;
            line = line.substr(a);

            // Eight characters was low enough to let a truncated answer's stubs through -- "The cost",
            // "The memory", "The memory of" -- which then sit in the prompt as if they meant something.
            // Four words is the floor for a thing worth remembering.
            if (line.size() < 8 || std::count(line.begin(), line.end(), ' ') < 3)
                continue;

            entry.text      = line;
            entry.createdAt = NowSeconds();
            out.push_back(std::move(entry));

            if (out.size() >= 12)    // sanity cap on one condensation pass
                break;
        }

        return out;
    }
}

// --------------------------------------------------------------------------

uint32_t Memory_EstimateTokens(const std::string& text)
{
    return static_cast<uint32_t>((text.size() + 3) / 4);
}

void Memory_Load()
{
    std::lock_guard<std::mutex> lock(g_mutex);
    g_state.clear();

    if (QueryResult result = CharacterDatabase.Query(
            "SELECT bot_guid, memory_text, importance, UNIX_TIMESTAMP(created_at) "
            "FROM mod_ollama_chat_memories ORDER BY importance DESC"))
    {
        uint32_t loaded = 0;
        do
        {
            Field* f = result->Fetch();
            BotMemoryEntry e;
            e.text       = f[1].Get<std::string>();
            e.importance = f[2].Get<uint8>();
            e.createdAt  = f[3].Get<uint64>();

            if (!e.text.empty())
            {
                g_state[f[0].Get<uint64>()].memories.push_back(std::move(e));
                ++loaded;
            }
        } while (result->NextRow());

        LOG_INFO("module.ollamachat", "[Ollama Chat] Loaded {} bot memories.", loaded);
    }

    if (QueryResult result = CharacterDatabase.Query(
            "SELECT bot_guid, other_guid, other_name, description, mentions, "
            "UNIX_TIMESTAMP(updated_at) FROM mod_ollama_chat_relationships"))
    {
        uint32_t loaded = 0;
        do
        {
            Field* f = result->Fetch();
            BotRelationship r;
            r.otherGuid   = f[1].Get<uint64>();
            r.otherName   = f[2].Get<std::string>();
            r.description = f[3].Get<std::string>();
            r.mentions    = f[4].Get<uint32>();
            r.updatedAt   = f[5].Get<uint64>();

            g_state[f[0].Get<uint64>()].relationships[r.otherGuid] = std::move(r);
            ++loaded;
        } while (result->NextRow());

        LOG_INFO("module.ollamachat", "[Ollama Chat] Loaded {} bot relationships.", loaded);
    }
}

namespace
{
    // Characters belonging to a real person, by lower-cased name -> account id. Playerbots hold accounts
    // too, so the test is the account, not the character: every bot account is the configured random-bot
    // prefix ("rndbot") and nothing else is. 5 rows here today, 157 bot accounts beside them.
    std::unordered_map<std::string, uint32_t> g_RealCharAccount;
    std::unordered_map<uint32_t, uint32_t>    g_AccountOfGuid;     // character guid (counter) -> account
    std::mutex                                g_HouseholdMutex;

    std::string Lower(std::string v)
    {
        for (char& c : v)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return v;
    }
}

void Memory_LoadHouseholds()
{
    std::string prefix = Lower(sConfigMgr->GetOption<std::string>(
        "AiPlayerbot.RandomBotAccountPrefix", "rndbot"));

    // Which accounts are a person's. Two tests, because neither alone is right here:
    //
    //   - not the random-bot prefix: 156 accounts on this realm are "rndbot<n>";
    //   - and has actually been signed into. This one catches the accounts that hold characters but no
    //     person: MERCHANTS (plans/17) owns ten level-1 traders who have never drawn breath, and the
    //     prefix test alone would have declared all ten of them people and held back any memory naming
    //     Tobble or Merrick. Playerbots never sets last_login -- 106 of them were online while this was
    //     measured and not one had a value -- so it separates furniture from people cleanly.
    //
    // A new player is unprotected until their first login, which costs nothing: nobody can hold a memory
    // about someone who has never been here.
    std::unordered_set<uint32_t> real;
    if (QueryResult result = LoginDatabase.Query(
            "SELECT id, username FROM account WHERE last_login IS NOT NULL"))
    {
        do
        {
            Field* f = result->Fetch();
            if (Lower(f[1].Get<std::string>()).rfind(prefix, 0) != 0)
                real.insert(f[0].Get<uint32_t>());
        } while (result->NextRow());
    }

    std::unordered_map<std::string, uint32_t> names;
    std::unordered_map<uint32_t, uint32_t>    accounts;
    if (QueryResult result = CharacterDatabase.Query("SELECT guid, name, account FROM characters"))
    {
        do
        {
            Field* f = result->Fetch();
            const uint32_t account = f[2].Get<uint32_t>();
            accounts[f[0].Get<uint32_t>()] = account;
            if (real.count(account))
                names[Lower(f[1].Get<std::string>())] = account;
        } while (result->NextRow());
    }

    {
        std::lock_guard<std::mutex> lock(g_HouseholdMutex);
        g_RealCharAccount = std::move(names);
        g_AccountOfGuid   = std::move(accounts);
    }

    LOG_INFO("module.ollamachat",
             "[Ollama Chat] {} characters on {} real accounts are protected from memory bleed.",
             g_RealCharAccount.size(), real.size());
}

// Would telling this memory here carry one person's doings to another person? A memory naming a character
// who belongs to a real account is held back unless someone of that household is here to hear it.
//
// Ranking cannot do this job: plan 31 §5 deliberately reserves a share of the prompt for a bot's own
// defining memories, unconditioned by context, and that share is exactly the hole a memory about an absent
// player would come through. Withholding has to be a gate, separate from the score. 140 of the 330 memories
// on this realm name one of the operator's five characters, held by 47 different bots.
//
// The test is a name in the text, which is crude -- it is what plan 31's tags replace. Crude is safe in this
// direction: a false positive withholds a memory, a false negative is only what happens today.
bool Memory_MayTell(const std::string& text, const std::unordered_set<uint32_t>& presentAccounts)
{
    std::lock_guard<std::mutex> lock(g_HouseholdMutex);
    if (g_RealCharAccount.empty())
        return true;

    const std::string hay = Lower(text);
    for (auto const& [name, account] : g_RealCharAccount)
    {
        if (presentAccounts.count(account))
            continue;                       // their own household is here; nothing is being carried
        size_t pos = 0;
        while ((pos = hay.find(name, pos)) != std::string::npos)
        {
            const bool startOk = (pos == 0) || !std::isalnum(static_cast<unsigned char>(hay[pos - 1]));
            const size_t end = pos + name.size();
            const bool endOk = (end >= hay.size()) || !std::isalnum(static_cast<unsigned char>(hay[end]));
            if (startOk && endOk)
                return false;
            pos = end;
        }
    }
    return true;
}

void Memory_Remember(uint64_t botGuid, const std::string& text, uint8_t importance)
{
    if (!g_MemoryEnable || botGuid == 0 || text.empty())
        return;

    BotMemoryEntry entry;
    entry.text       = text;
    entry.importance = importance < 1 ? 1 : (importance > 10 ? 10 : importance);
    entry.createdAt  = uint64_t(time(nullptr));

    std::lock_guard<std::mutex> lock(g_mutex);

    OllamaMemoryState& state = g_state[botGuid];
    state.memories.push_back(std::move(entry));

    // The cap is enforced on the condensation path only, and a bot that never
    // condenses would otherwise accumulate these without limit. Drop the least
    // important rather than the oldest: a trivial thought from a minute ago is
    // worth less than something that mattered yesterday.
    if (g_MemoryMaxPerBot > 0 && state.memories.size() > g_MemoryMaxPerBot)
    {
        std::sort(state.memories.begin(), state.memories.end(),
                  [](const BotMemoryEntry& a, const BotMemoryEntry& b)
                  {
                      if (a.importance != b.importance)
                          return a.importance > b.importance;
                      return a.createdAt > b.createdAt;
                  });
        state.memories.resize(g_MemoryMaxPerBot);
    }

    state.dirty = true;
}

void Memory_SaveAll()
{
    std::lock_guard<std::mutex> lock(g_mutex);

    for (auto& [botGuid, state] : g_state)
    {
        if (!state.dirty)
            continue;

        // The delete and the reinserts have to land together. As separate
        // async statements, a crash between them left the bot with no memories
        // at all, which is worse than a stale set.
        CharacterDatabaseTransaction trans = CharacterDatabase.BeginTransaction();

        trans->Append(SafeFormat(
            "DELETE FROM mod_ollama_chat_memories WHERE bot_guid = {}", botGuid));

        std::string values;
        for (const BotMemoryEntry& m : state.memories)
        {
            if (!values.empty())
                values += ',';

            values += SafeFormat("({}, '{}', {}, FROM_UNIXTIME({}))",
                                 botGuid, Escape(m.text), uint32_t(m.importance), m.createdAt);
        }

        if (!values.empty())
        {
            trans->Append("INSERT INTO mod_ollama_chat_memories "
                          "(bot_guid, memory_text, importance, created_at) VALUES " + values);
        }

        for (const auto& [otherGuid, r] : state.relationships)
        {
            if (r.description.empty())
                continue;

            // ON DUPLICATE KEY UPDATE rather than REPLACE INTO: REPLACE is a
            // delete plus an insert, so it rewrites the row and every index
            // entry even when nothing about the relationship changed.
            trans->Append(SafeFormat(
                "INSERT INTO mod_ollama_chat_relationships "
                "(bot_guid, other_guid, other_name, description, mentions, updated_at) "
                "VALUES ({}, {}, '{}', '{}', {}, FROM_UNIXTIME({})) "
                "ON DUPLICATE KEY UPDATE other_name = VALUES(other_name), "
                "description = VALUES(description), mentions = VALUES(mentions), "
                "updated_at = VALUES(updated_at)",
                botGuid, otherGuid, Escape(r.otherName), Escape(r.description),
                r.mentions, r.updatedAt ? r.updatedAt : NowSeconds()));
        }

        CharacterDatabase.CommitTransaction(trans);

        state.dirty = false;
    }
}

void Memory_ForgetBot(ObjectGuid botGuid)
{
    // Memories persist in the database; this only drops the in-memory copy for
    // a character who has logged out.
    std::lock_guard<std::mutex> lock(g_mutex);
    g_state.erase(botGuid.GetRawValue());
}

// --------------------------------------------------------------------------

void Memory_NoteExchange(uint64_t botGuid, uint64_t otherGuid,
                         const std::string& otherName,
                         const std::string& incomingMessage,
                         const std::string& botReply)
{
    if (!g_MemoryEnable || botGuid == 0)
        return;

    Player* bot = ObjectAccessor::FindPlayer(ObjectGuid(botGuid));
    if (!bot)
        return;

    // ---- relationship mention counting ---------------------------------
    if (g_RelationshipEnable && otherGuid != 0 && !otherName.empty())
    {
        const std::string combined = incomingMessage + " " + botReply;
        const uint32_t hits = 1 + CountMentions(combined, otherName);

        bool trigger = false;
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            OllamaMemoryState& state = g_state[botGuid];
            BotRelationship& rel = state.relationships[otherGuid];

            rel.otherGuid = otherGuid;
            if (rel.otherName.empty())
                rel.otherName = otherName;
            rel.mentions += hits;
            state.dirty = true;

            if (rel.mentions >= g_RelationshipMentionThreshold &&
                !state.relationshipPending[otherGuid])
            {
                state.relationshipPending[otherGuid] = true;
                rel.mentions = 0;      // reset the counter for the next revision
                trigger = true;
            }
        }

        if (trigger)
        {
            std::string prompt = Memory_BuildRelationshipPrompt(bot, otherGuid, otherName);
            if (!prompt.empty())
                OllamaDispatch_SubmitRelationship(botGuid, otherGuid, otherName, prompt);
            else
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                g_state[botGuid].relationshipPending[otherGuid] = false;
            }
        }
    }

    // ---- condensation ---------------------------------------------------
    if (g_MemoryHistoryTokenLimit == 0)
        return;

    if (HistoryTokens(botGuid) < g_MemoryHistoryTokenLimit)
        return;

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        OllamaMemoryState& state = g_state[botGuid];
        if (state.condensing)
            return;
        state.condensing = true;
    }

    std::string prompt = Memory_BuildCondensationPrompt(bot);
    if (prompt.empty())
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_state[botGuid].condensing = false;
        return;
    }

    OllamaDispatch_SubmitCondensation(botGuid, prompt);
}

// --------------------------------------------------------------------------

void Memory_LoadCompanions()
{
    // Which bots have ever stood in a company with a person. Same two tests the household gate uses to
    // tell a person from furniture: not the random-bot account prefix, and actually signed in once --
    // MERCHANTS (plans/17) owns ten level-1 traders who have never drawn breath.
    std::string prefix = Lower(sConfigMgr->GetOption<std::string>(
        "AiPlayerbot.RandomBotAccountPrefix", "rndbot"));

    std::unordered_set<uint32_t> realAccounts;
    if (QueryResult result = LoginDatabase.Query(
            "SELECT id, username FROM account WHERE last_login IS NOT NULL"))
    {
        do
        {
            Field* f = result->Fetch();
            if (Lower(f[1].Get<std::string>()).rfind(prefix, 0) != 0)
                realAccounts.insert(f[0].Get<uint32_t>());
        } while (result->NextRow());
    }

    std::unordered_set<uint32_t> realChars;
    if (QueryResult result = CharacterDatabase.Query("SELECT guid, account FROM characters"))
    {
        do
        {
            Field* f = result->Fetch();
            if (realAccounts.count(f[1].Get<uint32_t>()))
                realChars.insert(f[0].Get<uint32_t>());
        } while (result->NextRow());
    }

    // mod-ledger owns this table; if it is not installed the set simply starts empty and fills live as
    // bots are seen grouped with someone. A player's guid has no high part, so the raw guid is the low one.
    std::unordered_set<uint64_t> found;
    if (QueryResult result = CharacterDatabase.Query(
            "SELECT actor_guid, COALESCE(subject_guid, 0) FROM ledger_event WHERE event_type = 'group_join'"))
    {
        do
        {
            Field* f = result->Fetch();
            const uint32_t actor   = f[0].Get<uint32_t>();
            const uint32_t subject = f[1].Get<uint32_t>();
            if (subject && realChars.count(subject) && !realChars.count(actor))
                found.insert(actor);
            else if (subject && realChars.count(actor) && !realChars.count(subject))
                found.insert(subject);
        } while (result->NextRow());
    }

    const size_t n = found.size();
    {
        std::lock_guard<std::mutex> lock(g_CompanionMutex);
        g_Companions = std::move(found);
    }
    // Says which mode is actually live, because the count alone reads as a restriction even when the
    // restriction is off, and this line is the only thing anyone has to check it by.
    LOG_INFO("module.ollamachat",
             "[Ollama Chat] {} bots have travelled with a person; deed memories are {}.",
             n, g_MemoryEventCompanionsOnly ? "kept by them alone"
                                            : "kept by every bot in the world");
}


void Memory_NoteCompanion(uint64_t botGuid)
{
    if (botGuid == 0)
        return;

    std::lock_guard<std::mutex> lock(g_CompanionMutex);
    g_Companions.insert(botGuid);
}


bool Memory_IsCompanion(uint64_t botGuid)
{
    std::lock_guard<std::mutex> lock(g_CompanionMutex);
    return g_Companions.count(botGuid) != 0;
}


void Memory_NoteGameEvent(uint64_t botGuid, const std::string& line)
{
    if (!g_MemoryEnable || !g_MemoryEventEnable || botGuid == 0 || line.empty())
        return;

    // Optionally only a bot someone has actually travelled with. Left as a switch rather than a rule
    // because the cost of the alternative turned out to be affordable once the stale flush stopped
    // digesting near-empty buffers: it is the thin digests that were expensive, not the world being busy.
    if (g_MemoryEventCompanionsOnly && !Memory_IsCompanion(botGuid))
        return;

    Player* bot = ObjectAccessor::FindPlayer(ObjectGuid(botGuid));
    if (!bot)
        return;

    bool trigger = false;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        OllamaMemoryState& state = g_state[botGuid];

        if (state.eventBuffer.empty())
            state.eventFirstAt = NowSeconds();

        state.eventBuffer.push_back(line);

        // A long fight must not grow this without bound while a digest is in
        // flight: keep the most recent, because a stale deed matters least.
        const size_t hardCap = size_t(g_MemoryEventFlushCount ? g_MemoryEventFlushCount : 4) * 3;
        while (state.eventBuffer.size() > hardCap)
            state.eventBuffer.erase(state.eventBuffer.begin());

        if (!state.eventFlushing && g_MemoryEventFlushCount > 0 &&
            state.eventBuffer.size() >= g_MemoryEventFlushCount)
        {
            state.eventFlushing = true;
            trigger = true;
        }
    }

    if (!trigger)
        return;

    std::string prompt = Memory_BuildEventPrompt(bot);
    if (prompt.empty())
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_state[botGuid].eventFlushing = false;
        return;
    }

    OllamaDispatch_SubmitEventDigest(botGuid, prompt);
}

void Memory_FlushStaleEvents()
{
    if (!g_MemoryEnable || !g_MemoryEventEnable || g_MemoryEventFlushSeconds == 0)
        return;

    const uint64_t now = NowSeconds();

    // Claim the stale buffers under the lock, then build and submit outside it:
    // Memory_BuildEventPrompt takes the same mutex.
    std::vector<uint64_t> due;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        for (auto& [botGuid, state] : g_state)
        {
            if (state.eventFlushing || state.eventBuffer.empty())
                continue;
            if (now - state.eventFirstAt < g_MemoryEventFlushSeconds)
                continue;

            // A buffer this thin is dropped rather than digested, and this one test is what makes the
            // whole world affordable. Measured on the first night: 57 notable events became 284 model
            // calls, because every bot that had witnessed ANYTHING got a full generation once its five
            // minutes were up. The deeds were never the cost; flushing near-empty buffers was. It is also
            // where the quality went -- asked to remember one thing, the model padded to fill the quota,
            // which is why three quarters of those memories named no place and read like weather.
            if (state.eventBuffer.size() < g_MemoryEventFlushMinimum)
            {
                state.eventBuffer.clear();
                state.eventFirstAt = 0;
                continue;
            }

            state.eventFlushing = true;
            due.push_back(botGuid);
        }
    }

    for (const uint64_t botGuid : due)
    {
        Player* bot = ObjectAccessor::FindPlayer(ObjectGuid(botGuid));
        std::string prompt = bot ? Memory_BuildEventPrompt(bot) : "";

        if (prompt.empty())
        {
            // The bot has logged out, or there is no template. Release the
            // claim; the buffer keeps until it comes back.
            std::lock_guard<std::mutex> lock(g_mutex);
            g_state[botGuid].eventFlushing = false;
            continue;
        }

        OllamaDispatch_SubmitEventDigest(botGuid, prompt);
    }
}

// --------------------------------------------------------------------------

std::string Memory_BuildCondensationPrompt(Player* bot)
{
    if (!bot || g_MemoryCondensePrompt.empty())
        return "";

    const std::string history = RenderHistory(bot->GetGUID().GetRawValue());
    if (history.empty())
        return "";

    return SafeFormat(g_MemoryCondensePrompt,
                      fmt::arg("bot_name", bot->GetName()),
                      fmt::arg("history", history));
}

std::string Memory_BuildEventPrompt(Player* bot)
{
    if (!bot || g_MemoryEventPrompt.empty())
        return "";

    std::string events;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_state.find(bot->GetGUID().GetRawValue());
        if (it == g_state.end() || it->second.eventBuffer.empty())
            return "";

        for (const std::string& line : it->second.eventBuffer)
            events += " - " + line + "\n";
    }

    // The company is read here and not at buffer time: who you were with is a
    // fact about the outing, and the group is only safe to walk on this thread.
    std::string company;
    if (Group* group = bot->GetGroup())
    {
        for (GroupReference* ref = group->GetFirstMember(); ref; ref = ref->next())
        {
            Player* member = ref->GetSource();
            if (!member || member == bot || !member->IsInWorld())
                continue;
            if (!company.empty())
                company += ", ";
            company += member->GetName();
        }
    }
    // Not "no one; you were alone": asked who was with it, a model handed that phrasing back as something
    // to remember, and 131 of 892 memories were about solitude rather than about anything that happened.
    if (company.empty())
        company = "no one";

    return SafeFormat(g_MemoryEventPrompt,
                      fmt::arg("bot_name", bot->GetName()),
                      fmt::arg("company", company),
                      fmt::arg("events", events));
}

std::string Memory_BuildRelationshipPrompt(Player* bot, uint64_t otherGuid,
                                           const std::string& otherName)
{
    if (!bot || g_RelationshipUpdatePrompt.empty())
        return "";

    std::string existing;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_state.find(bot->GetGUID().GetRawValue());
        if (it != g_state.end())
        {
            auto relIt = it->second.relationships.find(otherGuid);
            if (relIt != it->second.relationships.end())
                existing = relIt->second.description;
        }
    }

    const std::string history = RenderHistory(bot->GetGUID().GetRawValue());

    return SafeFormat(g_RelationshipUpdatePrompt,
                      fmt::arg("bot_name", bot->GetName()),
                      fmt::arg("other_name", otherName),
                      fmt::arg("existing", existing.empty() ? "nothing yet" : existing),
                      fmt::arg("history", history));
}

// --------------------------------------------------------------------------

std::string Memory_BuildPromptSection(Player* bot, Player* about)
{
    if (!g_MemoryEnable || !bot)
        return "";

    const uint64_t botGuid   = bot->GetGUID().GetRawValue();
    const uint64_t aboutGuid = about ? about->GetGUID().GetRawValue() : 0;

    std::vector<BotMemoryEntry> memories;
    std::vector<BotRelationship> relationships;

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_state.find(botGuid);
        if (it == g_state.end())
            return "";

        memories = it->second.memories;
        for (const auto& [guid, rel] : it->second.relationships)
            if (!rel.description.empty())
                relationships.push_back(rel);
    }

    std::string out;

    // ---- relationships: whoever we are talking to comes first ----------
    if (g_RelationshipEnable && !relationships.empty())
    {
        std::sort(relationships.begin(), relationships.end(),
                  [aboutGuid](const BotRelationship& a, const BotRelationship& b)
                  {
                      if ((a.otherGuid == aboutGuid) != (b.otherGuid == aboutGuid))
                          return a.otherGuid == aboutGuid;
                      return a.updatedAt > b.updatedAt;
                  });

        std::string lines;
        uint32_t taken = 0;
        for (const BotRelationship& r : relationships)
        {
            if (g_RelationshipMaxPerPrompt > 0 && taken >= g_RelationshipMaxPerPrompt)
                break;
            lines += " - " + r.otherName + ": " + r.description + "\n";
            ++taken;
        }

        if (!lines.empty() && !g_RelationshipPromptTemplate.empty())
            out += SafeFormat(g_RelationshipPromptTemplate, fmt::arg("relationships", lines));
    }

    // ---- memories: most important first, inside a token budget ---------
    if (!memories.empty() && !g_MemoryPromptTemplate.empty())
    {
        std::sort(memories.begin(), memories.end(),
                  [](const BotMemoryEntry& a, const BotMemoryEntry& b)
                  {
                      if (a.importance != b.importance)
                          return a.importance > b.importance;
                      return a.createdAt > b.createdAt;
                  });

        // Whose doings may be spoken of here: the person being answered, and everyone standing with this
        // bot. Cheap -- a party is at most five, and the map lookup is a hash.
        std::unordered_set<uint32_t> present;
        auto note = [&present](Player* p)
        {
            if (!p)
                return;
            std::lock_guard<std::mutex> lock(g_HouseholdMutex);
            auto it = g_AccountOfGuid.find(p->GetGUID().GetCounter());
            if (it != g_AccountOfGuid.end())
                present.insert(it->second);
        };
        note(about);
        note(bot);
        if (Group* group = bot->GetGroup())
            for (GroupReference* ref = group->GetFirstMember(); ref; ref = ref->next())
                note(ref->GetSource());

        std::string lines;
        uint32_t used = 0;
        uint32_t withheld = 0;
        for (const BotMemoryEntry& m : memories)
        {
            const uint32_t cost = Memory_EstimateTokens(m.text) + 4;
            if (g_MemoryPromptTokenBudget > 0 && used + cost > g_MemoryPromptTokenBudget)
                continue;      // try the next, shorter one rather than stopping
            if (g_MemoryHouseholdGate && !Memory_MayTell(m.text, present))
            {
                ++withheld;
                continue;
            }
            lines += " - " + m.text + "\n";
            used += cost;
        }

        if (withheld && g_DebugEnabled)
            LOG_INFO("module.ollamachat", "[Ollama Chat] {} held back {} memories about absent households.",
                     bot->GetName(), withheld);

        if (!lines.empty())
            out += SafeFormat(g_MemoryPromptTemplate, fmt::arg("memories", lines));
    }

    return out;
}

// --------------------------------------------------------------------------
// Worker-side
// --------------------------------------------------------------------------

void Memory_RunCondensation(uint64_t botGuid, const std::string& prompt)
{
    OllamaApiResult api = QueryOllama(prompt, OllamaRequestKind::Sentiment);

    std::vector<BotMemoryEntry> fresh;
    if (api.ok)
        fresh = ParseMemories(api.text);

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        OllamaMemoryState& state = g_state[botGuid];
        state.condensing = false;

        if (!fresh.empty())
        {
            state.memories.insert(state.memories.end(), fresh.begin(), fresh.end());

            // Keep the most important, drop the rest.
            std::sort(state.memories.begin(), state.memories.end(),
                      [](const BotMemoryEntry& a, const BotMemoryEntry& b)
                      {
                          if (a.importance != b.importance)
                              return a.importance > b.importance;
                          return a.createdAt > b.createdAt;
                      });

            if (g_MemoryMaxPerBot > 0 && state.memories.size() > g_MemoryMaxPerBot)
                state.memories.resize(g_MemoryMaxPerBot);

            state.dirty = true;
        }
    }

    if (!api.ok)
    {
        if (g_DebugEnabled)
            LOG_INFO("module.ollamachat",
                     "[Ollama Chat] Memory condensation failed for bot {}: {}",
                     botGuid, api.error);
        return;      // keep the history; we will try again next time
    }

    // The history has been distilled; drop it so the window starts fresh.
    ClearHistory(botGuid);

    if (g_DebugEnabled)
        LOG_INFO("module.ollamachat",
                 "[Ollama Chat] Condensed history for bot {} into {} memories.",
                 botGuid, fresh.size());
}

void Memory_RunEventDigest(uint64_t botGuid, const std::string& prompt)
{
    OllamaApiResult api = QueryOllama(prompt, OllamaRequestKind::Sentiment);

    std::vector<BotMemoryEntry> fresh;
    if (api.ok)
        fresh = ParseMemories(api.text);

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        OllamaMemoryState& state = g_state[botGuid];
        state.eventFlushing = false;

        if (api.ok)
        {
            // Distilled or not, these deeds have had their turn: keeping them
            // would re-digest the same fight on the next flush.
            state.eventBuffer.clear();
            state.eventFirstAt = 0;
        }

        if (!fresh.empty())
        {
            state.memories.insert(state.memories.end(), fresh.begin(), fresh.end());

            // Keep the most important, drop the rest.
            std::sort(state.memories.begin(), state.memories.end(),
                      [](const BotMemoryEntry& a, const BotMemoryEntry& b)
                      {
                          if (a.importance != b.importance)
                              return a.importance > b.importance;
                          return a.createdAt > b.createdAt;
                      });

            if (g_MemoryMaxPerBot > 0 && state.memories.size() > g_MemoryMaxPerBot)
                state.memories.resize(g_MemoryMaxPerBot);

            state.dirty = true;
        }
    }

    if (!api.ok)
    {
        if (g_DebugEnabled)
            LOG_INFO("module.ollamachat",
                     "[Ollama Chat] Event digest failed for bot {}: {}",
                     botGuid, api.error);
        return;      // keep the deeds; we will try again on the next one
    }

    if (g_DebugEnabled)
        LOG_INFO("module.ollamachat",
                 "[Ollama Chat] Digested deeds for bot {} into {} memories.",
                 botGuid, fresh.size());
}

void Memory_RunRelationshipUpdate(uint64_t botGuid, uint64_t otherGuid,
                                  const std::string& otherName,
                                  const std::string& prompt)
{
    OllamaApiResult api = QueryOllama(prompt, OllamaRequestKind::Sentiment);

    std::string description;
    if (api.ok)
    {
        description = api.text;

        // One sentence, no markup, bounded length.
        for (char& c : description)
            if (c == '\n' || c == '\r' || c == '\t')
                c = ' ';

        const size_t cut = description.find_first_of(" \t");
        (void)cut;

        if (description.size() > g_RelationshipMaxLength)
        {
            description.resize(g_RelationshipMaxLength);
            const size_t lastSpace = description.find_last_of(' ');
            if (lastSpace != std::string::npos && lastSpace > g_RelationshipMaxLength / 2)
                description.erase(lastSpace);
        }
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    OllamaMemoryState& state = g_state[botGuid];
    state.relationshipPending[otherGuid] = false;

    if (description.empty())
        return;

    BotRelationship& rel = state.relationships[otherGuid];
    rel.otherGuid   = otherGuid;
    rel.otherName   = otherName;
    rel.description = std::move(description);
    rel.updatedAt   = NowSeconds();
    state.dirty     = true;

    if (g_DebugEnabled)
        LOG_INFO("module.ollamachat",
                 "[Ollama Chat] Bot {} relationship with {} updated: {}",
                 botGuid, otherName, rel.description);
}
