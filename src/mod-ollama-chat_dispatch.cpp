#include "mod-ollama-chat_dispatch.h"
#include "mod-ollama-chat_api.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat_expression.h"
#include "mod-ollama-chat_governor.h"
#include "mod-ollama-chat_memory.h"
#include "mod-ollama-chat_response.h"
#include "mod-ollama-chat_roleplay.h"
#include "mod-ollama-chat_sentiment.h"
#include "mod-ollama-chat-utilities.h"
#include "mod-ollama-chat_world.h"

#include <nlohmann/json.hpp>
#include <random>

#include "CellImpl.h"
#include "Channel.h"
#include "ChannelMgr.h"
#include "GridNotifiers.h"
#include "GridNotifiersImpl.h"
#include "Group.h"
#include "Guild.h"
#include "Log.h"
#include "ObjectAccessor.h"
#include "Player.h"

#include "PlayerbotAI.h"
#include "PlayerbotMgr.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

// Local patch (custom-wow): mod-ledger records channel replies, which bypass the
// OnPlayerCanUseChat hooks. Weak so this links without it.
void LedgerRecordBotChat(Player* bot, uint32 type, std::string const& msg, Channel* channel) __attribute__((weak));

namespace
{
    using Clock = std::chrono::steady_clock;

    enum class TaskType : uint8_t { ChatReply, Sentiment, Condense, Relationship, Classify, HeldTongue,
                                    EventDigest };

    struct Task
    {
        TaskType          type = TaskType::ChatReply;
        OllamaChatRequest request;

        // Sentiment-only payload.
        uint64_t    sentimentBotGuid    = 0;
        uint64_t    sentimentPlayerGuid = 0;
        std::string sentimentMessage;
        std::string sentimentPrompt;

        // Memory / relationship payload.
        uint64_t    memoryBotGuid   = 0;
        uint64_t    memoryOtherGuid = 0;
        std::string memoryOtherName;
        std::string memoryPrompt;

        // Addressee-pass payload: everything needed to submit the real replies
        // once the answer comes back, since ProcessChat's locals are gone by then.
        OllamaAddresseeRequest addressee;

        // Held-tongue payload: who stayed quiet, and what they stayed quiet about.
        OllamaHeldTongueRequest heldTongue;
    };

    struct Completion
    {
        OllamaChatRequest request;
        std::string       text;
        uint32_t          emoteId = 0;
        Clock::time_point deliverAt;

        // An addressee answer is not a spoken line. It comes back through the
        // same queue so it lands on the world thread, but the drain hands it to
        // the resolver instead of to Deliver -- nothing here is ever said.
        bool                   isClassify = false;
        OllamaAddresseeRequest addressee;

        // Nor is a held tongue. It comes back the same way, for the same
        // reason: the emote and the memory write are both world-thread work.
        bool                    isHeldTongue = false;
        OllamaHeldTongueRequest heldTongue;
    };

    // --- shared state -----------------------------------------------------

    std::mutex              g_queueMutex;
    std::condition_variable g_queueCv;
    std::deque<Task>        g_queue;
    bool                    g_running = false;

    std::mutex             g_doneMutex;
    std::deque<Completion> g_done;

    std::vector<std::thread> g_workers;

    std::atomic<uint32_t> g_inFlight{ 0 };
    std::atomic<uint64_t> g_totalSubmitted{ 0 };
    std::atomic<uint64_t> g_totalDelivered{ 0 };
    std::atomic<uint64_t> g_droppedQueueFull{ 0 };
    std::atomic<uint64_t> g_droppedEmpty{ 0 };
    std::atomic<uint64_t> g_droppedGovernor{ 0 };
    std::atomic<uint64_t> g_totalFailed{ 0 };

    std::mutex  g_errorMutex;
    std::string g_lastError;

    // --- held-tongue emotes, deferred -------------------------------------
    //
    // The cheap lane answers a held tongue in well under a second, while the
    // reply it defers to takes about three. Firing the emote on arrival
    // therefore announced "holds their tongue and lets X speak" BEFORE X had
    // said anything -- all four emotes in the 2026-09-17 playtest landed about
    // three seconds early, which telegraphed the answer and inverted cause and
    // effect.
    //
    // So the emote waits for the line it names. All of this is world-thread
    // only -- ResolveHeldTongue, Deliver and OllamaDispatch_Update all run
    // there -- which is why none of it is locked.

    struct PendingEmote
    {
        uint64_t          botGuid = 0;
        std::string       speakerName;   // the line being waited for
        std::string       scopeKey;
        std::string       line;
        Clock::time_point giveUpAt;
    };

    std::vector<PendingEmote> g_pendingEmotes;

    // scopeKey + '|' + name -> when that person last spoke there, so a held
    // tongue that resolves AFTER the reply it names still fires rather than
    // waiting for a line that has already been said.
    std::unordered_map<std::string, Clock::time_point> g_recentSpeakers;

    std::atomic<uint64_t> g_heldTongueDeferred{ 0 };
    std::atomic<uint64_t> g_heldTongueFired{ 0 };
    std::atomic<uint64_t> g_heldTongueAbandoned{ 0 };
    std::atomic<uint64_t> g_tailsTrimmed{ 0 };

    std::string SpeakerKey(const std::string& scopeKey, const std::string& name)
    {
        return scopeKey + "|" + name;
    }

    // Re-resolves the bot: the wait is seconds long, and in that window it can
    // log out, die or leave exactly as a reply in flight can.
    void EmitHeldTongue(uint64_t botGuid, const std::string& line)
    {
        Player* bot = ObjectAccessor::FindConnectedPlayer(ObjectGuid(botGuid));
        if (!bot || !bot->IsInWorld() || !bot->IsAlive())
            return;

        bot->TextEmote(line);
        ++g_heldTongueFired;
    }

    bool SpokeRecently(const std::string& scopeKey, const std::string& name)
    {
        auto it = g_recentSpeakers.find(SpeakerKey(scopeKey, name));
        if (it == g_recentSpeakers.end())
            return false;

        return std::chrono::duration_cast<std::chrono::seconds>(
                   Clock::now() - it->second).count() <=
               int64_t(g_HeldTongueEmoteWaitSeconds);
    }

    // A line just landed here. Release every emote that was waiting for it.
    void NoteSpoken(const std::string& scopeKey, const std::string& speakerName)
    {
        g_recentSpeakers[SpeakerKey(scopeKey, speakerName)] = Clock::now();

        for (auto it = g_pendingEmotes.begin(); it != g_pendingEmotes.end(); )
        {
            if (it->scopeKey == scopeKey && it->speakerName == speakerName)
            {
                EmitHeldTongue(it->botGuid, it->line);
                it = g_pendingEmotes.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    void RecordError(const std::string& what)
    {
        ++g_totalFailed;
        std::lock_guard<std::mutex> lock(g_errorMutex);
        g_lastError = what;
    }

    // --- worker -----------------------------------------------------------

    void RunChatTask(const Task& task)
    {
        OllamaApiResult api = QueryOllama(task.request.prompt, task.request.kind);

        if (!api.ok)
        {
            RecordError(api.error);
            return;
        }

        uint32_t emoteId = 0;
        std::string text = ProcessLlmResponse(api.text, task.request.botName, &emoteId);
        if (task.request.maxWords > 0)
            text = ClampReplyWords(text, task.request.maxWords);

        // Roleplay mode rejects lines carrying out-of-world vocabulary rather
        // than mangling the sentence around the offending word.
        if (!text.empty())
        {
            std::string filtered = Roleplay_FilterMetaTerms(text);
            if (filtered.empty() && !text.empty() && g_DebugEnabled)
            {
                LOG_INFO("module.ollamachat",
                         "[Ollama Chat] Bot {} reply rejected by roleplay filter: '{}'",
                         task.request.botName, text);
            }
            text = std::move(filtered);
        }

        if (text.empty())
        {
            ++g_droppedEmpty;
            if (g_DebugEnabled)
                LOG_INFO("module.ollamachat",
                         "[Ollama Chat] Bot {} produced nothing usable after cleanup.",
                         task.request.botName);
            return;
        }

        Completion completion;
        completion.request = task.request;
        completion.text    = std::move(text);
        completion.emoteId = emoteId;

        uint32_t delayMs = 0;
        if (g_EnableTypingSimulation)
        {
            delayMs = g_TypingSimulationBaseDelay +
                      static_cast<uint32_t>(completion.text.length()) * g_TypingSimulationDelayPerChar;
            if (g_TypingSimulationMaxDelay > 0 && delayMs > g_TypingSimulationMaxDelay)
                delayMs = g_TypingSimulationMaxDelay;
        }

        // Whatever the caller asked to be held back on top of that -- the group
        // branch spacing its speakers, so a party answering together arrives as
        // several people rather than all at once.
        delayMs += task.request.extraDelayMs;

        // A delay, not a sleep. The old code held a whole thread hostage here.
        completion.deliverAt = Clock::now() + std::chrono::milliseconds(delayMs);

        std::lock_guard<std::mutex> lock(g_doneMutex);
        g_done.push_back(std::move(completion));
    }

    // The addressee pass, worker half: ask the cheap lane who the line was for.
    // Nothing here is ever spoken, so there is no response cleanup, no roleplay
    // filter and no typing delay -- the answer is a label, not a voice.
    void RunClassifyTask(const Task& task)
    {
        OllamaApiResult api = QueryOllama(task.request.prompt, OllamaRequestKind::Classify);

        Completion completion;
        completion.request    = task.request;
        completion.isClassify = true;
        completion.addressee  = task.addressee;
        completion.deliverAt  = Clock::now();

        if (api.ok)
        {
            completion.text = std::move(api.text);
        }
        else
        {
            // Worth recording, not worth losing the line over: an empty answer
            // makes the resolver fall back to letting the candidates speak,
            // which is what would have happened without the pass at all.
            RecordError(api.error);
        }

        std::lock_guard<std::mutex> lock(g_doneMutex);
        g_done.push_back(std::move(completion));
    }

    // A held tongue, worker half. Asks the cheap lane what a passed-over bot
    // kept to itself and how much it mattered, in one answer. Nothing here is
    // spoken, so there is no cleanup, no filter and no typing delay.
    void RunHeldTongueTask(const Task& task)
    {
        OllamaApiResult api = QueryOllama(task.request.prompt, OllamaRequestKind::Classify);

        Completion completion;
        completion.request      = task.request;
        completion.isHeldTongue = true;
        completion.heldTongue   = task.heldTongue;
        completion.deliverAt    = Clock::now();

        if (api.ok)
            completion.text = std::move(api.text);
        else
            RecordError(api.error);   // no thought, no emote; the bot stays quiet

        std::lock_guard<std::mutex> lock(g_doneMutex);
        g_done.push_back(std::move(completion));
    }

    void RunSentimentTask(const Task& task)
    {
        // Sentiment touches only mutex-guarded in-memory state and async DB
        // writes, so it completes here rather than round-tripping to the
        // world thread.
        ApplySentimentAnalysis(task.sentimentBotGuid, task.sentimentPlayerGuid,
                               task.sentimentMessage, task.sentimentPrompt);
    }

    void WorkerLoop()
    {
        for (;;)
        {
            Task task;

            {
                std::unique_lock<std::mutex> lock(g_queueMutex);
                g_queueCv.wait(lock, [] { return !g_running || !g_queue.empty(); });

                if (!g_running && g_queue.empty())
                    return;

                task = std::move(g_queue.front());
                g_queue.pop_front();
            }

            ++g_inFlight;

            try
            {
                switch (task.type)
                {
                    case TaskType::Sentiment:
                        RunSentimentTask(task);
                        break;
                    case TaskType::Condense:
                        Memory_RunCondensation(task.memoryBotGuid, task.memoryPrompt);
                        break;
                    case TaskType::Relationship:
                        Memory_RunRelationshipUpdate(task.memoryBotGuid, task.memoryOtherGuid,
                                                     task.memoryOtherName, task.memoryPrompt);
                        break;
                    case TaskType::Classify:
                        RunClassifyTask(task);
                        break;
                    case TaskType::HeldTongue:
                        RunHeldTongueTask(task);
                        break;
                    case TaskType::EventDigest:
                        Memory_RunEventDigest(task.memoryBotGuid, task.memoryPrompt);
                        break;
                    default:
                        RunChatTask(task);
                        break;
                }
            }
            catch (const std::exception& e)
            {
                RecordError(e.what());
                LOG_ERROR("module.ollamachat", "[Ollama Chat] Worker exception: {}", e.what());
            }
            catch (...)
            {
                RecordError("unknown exception");
                LOG_ERROR("module.ollamachat", "[Ollama Chat] Unknown worker exception.");
            }

            --g_inFlight;
        }
    }

    // --- delivery (world thread only) -------------------------------------

    Channel* ResolveChannel(Player* bot, const std::string& channelName)
    {
        if (!bot || channelName.empty())
            return nullptr;

        ChannelMgr* mgr = ChannelMgr::forTeam(bot->GetTeamId());
        if (!mgr)
            return nullptr;

        return mgr->GetChannel(channelName, bot);
    }

    bool AnyoneInRange(Player* bot, float distance)
    {
        if (!bot || !bot->IsInWorld() || distance <= 0.0f)
            return false;

        // Grid search: this runs on every Say/Yell delivery, and walking every
        // online character to answer "is anyone standing near me" is the kind
        // of thing that adds up on a bot-heavy realm.
        std::list<Player*> found;
        Acore::AnyPlayerInObjectRangeCheck check(bot, distance, false, true);
        Acore::PlayerListSearcher<Acore::AnyPlayerInObjectRangeCheck> searcher(bot, found, check);
        Cell::VisitObjects(bot, searcher, distance);

        for (Player* other : found)
            if (other && other != bot && other->IsInWorld())
                return true;

        return false;
    }

    // Returns true when the line actually went out.
    bool RouteMessage(Player* bot, PlayerbotAI* botAI, const Completion& c,
                      const OllamaWorldSnapshot& world,
                      Channel*& outChannel)
    {
        outChannel = nullptr;

        switch (c.request.source)
        {
            case SRC_GENERAL_LOCAL:
            {
                Channel* channel = ResolveChannel(bot, c.request.channelName);
                if (!channel || !bot->IsInChannel(channel))
                    return false;

                // Checked before generating too; re-checked because the only
                // human in the channel can leave during the LLM round trip.
                if (!world.RealPlayerInChannel(channel))
                    return false;

                channel->Say(bot->GetGUID(), c.text, LANG_UNIVERSAL);
                if (LedgerRecordBotChat)
                    LedgerRecordBotChat(bot, CHAT_MSG_CHANNEL, c.text, channel);
                outChannel = channel;
                return true;
            }

            // Guild, party and raid are re-checked here for the same reason
            // say and yell always were: the audience is validated at submit
            // time, and an LLM round trip is seconds long. Whoever the bot was
            // talking to can log out, leave the guild or drop group in that
            // window, and without this the bot announces to an empty channel.
            case SRC_GUILD_LOCAL:
            case SRC_OFFICER_LOCAL:
                if (g_DisableForGuild || !bot->GetGuild())
                    return false;
                if (!world.GuildHasRealPlayer(bot->GetGuildId()))
                    return false;
                return botAI->SayToGuild(c.text);

            // A company of bots is its own audience (plans/31 §19). The re-check above exists because an
            // LLM round trip is seconds long and the audience can leave inside it -- but it carried the
            // old assumption that a party is only worth speaking to when a person is in it. `c41e12e`
            // fixed that assumption in the reply path (`handler.cpp`) and never here, so every line a
            // bot-only company generated was thrown away at the last mile: 5 submitted, 0 delivered, and
            // no counter moved, because this return is the one drop in Deliver that tallies nothing.
            case SRC_PARTY_LOCAL:
                if (g_DisableForParty || !bot->GetGroup())
                    return false;
                if (!OllamaGroupHasRealPlayer(bot) &&
                    !(g_PartyChatterEnable && bot->GetGroup()->GetMembersCount() >= 2))
                    return false;
                return botAI->SayToParty(c.text);

            case SRC_RAID_LOCAL:
                if (g_DisableForParty || !bot->GetGroup())
                    return false;
                if (!OllamaGroupHasRealPlayer(bot) &&
                    !(g_PartyChatterEnable && bot->GetGroup()->GetMembersCount() >= 2))
                    return false;
                return botAI->SayToRaid(c.text);

            case SRC_YELL_LOCAL:
                if (g_DisableForSayYell || !AnyoneInRange(bot, g_YellDistance))
                    return false;
                return botAI->Yell(c.text);

            case SRC_WHISPER_LOCAL:
            {
                Player* target = ObjectAccessor::FindConnectedPlayer(ObjectGuid(c.request.targetGuid));
                if (!target)
                    return false;
                return botAI->Whisper(c.text, target->GetName());
            }

            case SRC_SAY_LOCAL:
            default:
                if (g_DisableForSayYell || !AnyoneInRange(bot, g_SayDistance))
                    return false;
                return botAI->Say(c.text);
        }
    }

    // Is `thought` just `said` handed back? Compared on words rather than characters so that punctuation,
    // case and a dropped article do not hide it.
    bool IsEcho(const std::string& thought, const std::string& said)
    {
        auto words = [](const std::string& v)
        {
            std::vector<std::string> out;
            std::string cur;
            for (char c : v)
            {
                if (std::isalnum(static_cast<unsigned char>(c)))
                    cur += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                else if (!cur.empty())
                {
                    out.push_back(cur);
                    cur.clear();
                }
            }
            if (!cur.empty())
                out.push_back(cur);
            return out;
        };

        const std::vector<std::string> a = words(thought);
        const std::vector<std::string> b = words(said);
        if (a.size() < 4 || b.empty())
            return false;

        size_t shared = 0;
        for (const std::string& w : a)
            if (std::find(b.begin(), b.end(), w) != b.end())
                ++shared;

        return shared * 4 >= a.size() * 3;      // three quarters of it came from them
    }

    // A held tongue, world-thread half. Reads {"thought": "...", "weight": N}
    // and does two things with it: the bot remembers the thought, and if it
    // weighed heavily enough, the room sees that something was withheld.
    //
    // The thought itself is never displayed -- only the fact of it -- so the
    // text never has to be safe to show.
    void ResolveHeldTongue(const Completion& c)
    {
        const OllamaHeldTongueRequest& h = c.heldTongue;

        if (c.text.empty())
            return;                 // lane down: no thought, no emote

        std::string thought;
        bool        mattered = false;

        const size_t open  = c.text.find('{');
        const size_t close = c.text.rfind('}');
        if (open == std::string::npos || close == std::string::npos || close <= open)
            return;

        try
        {
            const nlohmann::json j = nlohmann::json::parse(c.text.substr(open, close - open + 1));

            const auto t = j.find("thought");
            if (t != j.end() && t->is_string())
                thought = t->get<std::string>();

            // A yes or no, not a number. Some models answer the boolean as a
            // string, so both forms are accepted; anything else leaves it false
            // and the thought is merely remembered.
            const auto m = j.find("mattered");
            if (m != j.end())
            {
                if (m->is_boolean())
                {
                    mattered = m->get<bool>();
                }
                else if (m->is_string())
                {
                    const std::string s = m->get<std::string>();
                    mattered = (s == "true" || s == "True" || s == "yes" || s == "Yes");
                }
            }
        }
        catch (const std::exception&)
        {
            return;                 // unparseable: as if it never happened
        }

        if (thought.empty())
            return;

        // Only what mattered is kept. This path was writing every swallowed half-thought into the same
        // store the bot reads back as things it knows, and because condensation could never run
        // (plans/30 §4) it was the ONLY thing in there: 41 of the 48 memories on the realm after seven
        // days of play were "I kept my doubts to myself" or "I did not share my thoughts on the matter".
        // A bot whose whole remembered life is a list of times it said nothing has been taught to say
        // nothing. The ones that mattered are worth keeping; the rest were never worth a row.
        if (!mattered)
            return;

        // And never keep a parrot. Asked what it swallowed, a model will sometimes answer by repeating the
        // line that prompted the question, and that sentence then sits in the bot's memory until it says it
        // back to the person who said it first. Traced on 2026-09-20: a player said "The silence is
        // temporary, the sounds of battle will ring out too"; a bot stored it verbatim at 15:23:30 and
        // said it back to them at 15:47:47, then again eight seconds later.
        if (IsEcho(thought, h.message))
            return;

        // This is the half the player never sees, and the one that gives a swallowed line an effect on the
        // world: it sorts into the bot's prompt and colours what it says later.
        Memory_Remember(h.botGuid, thought, 7);

        if (!mattered || g_HeldTongueEmote.empty())
            return;

        // A custom string rather than a TEXT_EMOTE_* id: there is no "holds
        // tongue" emote in 3.3.5, and the stock table would force a compromise
        // like "ponders".
        const std::string line = SafeFormat(g_HeldTongueEmote,
                                            fmt::arg("speaker_name", h.speakerName));
        if (line.empty() || line == "[Format Error]")
            return;

        // The emote follows the line it defers to instead of racing ahead of
        // it. If that speaker has already been heard here -- this answer came
        // back late, or their reply was unusually quick -- there is nothing
        // left to wait for.
        if (SpokeRecently(h.scopeKey, h.speakerName))
        {
            EmitHeldTongue(h.botGuid, line);
        }
        else
        {
            PendingEmote p;
            p.botGuid     = h.botGuid;
            p.speakerName = h.speakerName;
            p.scopeKey    = h.scopeKey;
            p.line        = line;
            p.giveUpAt    = Clock::now() + std::chrono::seconds(g_HeldTongueEmoteWaitSeconds);

            g_pendingEmotes.push_back(std::move(p));
            ++g_heldTongueDeferred;
        }

        if (g_DebugEnabled)
            LOG_INFO("module.ollamachat",
                     "[Ollama Chat] {} held their tongue (mattered={}): '{}'",
                     h.botName, mattered, thought);
    }

    // The addressee pass, world-thread half. Reads {"to":["name", ...]} out of
    // the lane's answer and submits the real replies for whoever was addressed.
    //
    // Every failure here falls back to letting the candidates speak. The pass
    // may cost a call and change nothing; it must never turn a line into
    // silence, which would read as the bots being broken.
    void ResolveAddressee(const Completion& c)
    {
        const OllamaAddresseeRequest& a = c.addressee;

        Player* sender = ObjectAccessor::FindConnectedPlayer(ObjectGuid(a.senderGuid));
        if (!sender)
            return;

        // ASCII-only fold: 3.3.5 character names are ASCII, and this avoids
        // dragging locale handling into the dispatcher.
        auto sameName = [](const std::string& x, const std::string& y)
        {
            if (x.size() != y.size())
                return false;
            for (size_t i = 0; i < x.size(); ++i)
            {
                char cx = x[i], cy = y[i];
                if (cx >= 'A' && cx <= 'Z') cx = static_cast<char>(cx + 32);
                if (cy >= 'A' && cy <= 'Z') cy = static_cast<char>(cy + 32);
                if (cx != cy)
                    return false;
            }
            return true;
        };

        std::vector<std::string> named;
        bool parsed        = false;
        bool groupDirected = false;

        // The model is asked for bare JSON but may wrap it in a sentence, so
        // take the outermost braces rather than trusting the whole string.
        const size_t open  = c.text.find('{');
        const size_t close = c.text.rfind('}');
        if (open != std::string::npos && close != std::string::npos && close > open)
        {
            try
            {
                const nlohmann::json j = nlohmann::json::parse(c.text.substr(open, close - open + 1));
                const auto to = j.find("to");
                if (to != j.end() && to->is_array())
                {
                    for (const auto& n : *to)
                        if (n.is_string())
                            named.push_back(n.get<std::string>());
                    parsed = true;
                }

                // "Aimed at all of them" (plan 25 item 59). A separate key
                // rather than a new shape for `to`, so a model that ignores the
                // instruction still answers the old way and still parses: an
                // answer without it behaves exactly as it did before this
                // branch existed.
                const auto grp = j.find("group");
                if (grp != j.end())
                {
                    if (grp->is_boolean())
                    {
                        groupDirected = grp->get<bool>();
                        parsed        = true;
                    }
                    else if (grp->is_string())
                    {
                        const std::string s = grp->get<std::string>();
                        groupDirected = (s == "true" || s == "True" || s == "yes" || s == "Yes");
                        parsed        = true;
                    }
                }

                // Some models answer the group case by naming everyone instead,
                // or by writing {"to":"all"}. Read both as the group branch
                // rather than as three invented names.
                if (!groupDirected && to != j.end() && to->is_string())
                {
                    const std::string s = to->get<std::string>();
                    groupDirected = (s == "all" || s == "All" || s == "everyone" ||
                                     s == "Everyone" || s == "group" || s == "Group");
                    if (groupDirected)
                        parsed = true;
                }
            }
            catch (const std::exception&)
            {
                parsed = false;
            }
        }

        std::vector<size_t> speakers;

        std::random_device rd;
        std::mt19937       gen(rd());

        // The random MaxBotsToPick cut. It used to run in ProcessChat BEFORE
        // this pass, which meant the one bot the person was mid-conversation
        // with could be thrown away at random before anything got to choose
        // (plan 25 item 54). It runs here now, on whatever this resolver
        // actually picked -- so the pass always sees every candidate that
        // passed its roll, and the fallback below is still never wider than it
        // was before the pass existed.
        auto cutToMax = [&](std::vector<size_t>& v)
        {
            if (g_MaxBotsToPick == 0 || v.size() <= g_MaxBotsToPick)
                return;

            std::shuffle(v.begin(), v.end(), gen);
            std::uniform_int_distribution<uint32_t> howMany(1, g_MaxBotsToPick);
            v.resize(howMany(gen));
        };

        auto takeAll = [&]()
        {
            speakers.clear();
            for (size_t i = 0; i < a.candidateGuids.size(); ++i)
                speakers.push_back(i);
            cutToMax(speakers);
        };

        // Where the thread holder sits in the candidate list, if it is still
        // one of them.
        size_t holderIdx = SIZE_MAX;
        if (a.holderGuid)
        {
            for (size_t i = 0; i < a.candidateGuids.size(); ++i)
            {
                if (a.candidateGuids[i] == a.holderGuid)
                {
                    holderIdx = i;
                    break;
                }
            }
        }

        // How many of `speakers` may actually speak. 0 means all of them.
        uint32_t speakerCap = 0;

        if (!parsed)
        {
            takeAll();
        }
        else if (groupDirected)
        {
            // Item 59. A line to the whole party used to be the LEAST likely to
            // draw more than one answer: group-directed and nobody-directed
            // both arrived as {"to":[]}, and that meant one voice picked at
            // random, so addressing everyone made bots less likely to answer.
            //
            // Capped rather than open. The note asks for all of them, but the
            // pile-on this pass was built to remove is the other failure --
            // 2.50 replies per line before it, 1.06 after -- and five parallel
            // generations at 3.4-10.7 s apiece would queue behind four workers.
            // The bots passed over here hold their tongue, which is what keeps
            // the rest of the party present without speaking over it.
            for (size_t i = 0; i < a.candidateGuids.size(); ++i)
                speakers.push_back(i);

            std::shuffle(speakers.begin(), speakers.end(), gen);

            const uint32_t cap = std::max<uint32_t>(1, g_AddresseeGroupSpeakers);
            const uint32_t low = std::min<uint32_t>(2, cap);
            std::uniform_int_distribution<uint32_t> howMany(low, cap);

            speakerCap = howMany(gen);
        }
        else if (named.empty())
        {
            // Items 54 and 63. An empty verdict means the line named nobody --
            // not that it was meant for nobody. If this person is mid-exchange
            // with a bot here, that bot answers: "Exactly", or "Are you an
            // engineer?", belongs to whoever just spoke, and handing it to a
            // random voice is what made the operator correct himself thirty
            // seconds later. Random only when there is no thread at all.
            speakerCap = 1;

            if (holderIdx != SIZE_MAX)
            {
                speakers.push_back(holderIdx);
            }
            else if (!a.candidateGuids.empty())
            {
                std::uniform_int_distribution<size_t> pick(0, a.candidateGuids.size() - 1);
                speakers.push_back(pick(gen));
            }
        }
        else
        {
            speakerCap = a.maxSpeakers;

            for (const std::string& want : named)
            {
                for (size_t i = 0; i < a.candidateNames.size(); ++i)
                {
                    if (!sameName(a.candidateNames[i], want))
                        continue;

                    bool already = false;
                    for (size_t s : speakers)
                        if (s == i)
                            already = true;
                    if (!already)
                        speakers.push_back(i);
                    break;
                }
            }

            // Every name it gave was invented. Treat that as no answer at all.
            if (speakers.empty())
                takeAll();
        }

        uint32_t            spoken = 0;
        std::vector<size_t> spokeIdx;
        std::string         firstSpeakerName;

        for (size_t idx : speakers)
        {
            if (idx >= a.candidateGuids.size())
                continue;

            Player* bot = ObjectAccessor::FindConnectedPlayer(ObjectGuid(a.candidateGuids[idx]));
            if (!bot || !bot->IsInWorld() || !bot->IsAlive())
                continue;

            Channel* channel = a.channelId ? OllamaResolveZoneChannel(bot, a.channelId) : nullptr;

            // Only the group branch spaces its speakers. Everywhere else one
            // bot answers, and there is nothing to space it against.
            const uint32_t extraDelayMs =
                groupDirected ? spoken * g_AddresseeGroupStaggerMs : 0;

            if (OllamaSubmitBotReply(bot, sender, a.msg, a.trimmedMsg, a.source, channel,
                                     a.chainDepth, a.scopeKey, a.senderIsBot, extraDelayMs))
            {
                ++spoken;
                spokeIdx.push_back(idx);
                if (firstSpeakerName.empty())
                    firstSpeakerName = bot->GetName();
                if (speakerCap > 0 && spoken >= speakerCap)
                    break;
            }
        }

        // The one bot this pass chose was refused by the governor -- a cooldown,
        // a rate limit, an empty prompt. Handing the line to the next candidate
        // is better than the silence that used to follow, which reads as the
        // bots being broken rather than as one of them being busy (plan 25 §21).
        //
        // Only where the pass actually chose: on the fallback path every
        // candidate was tried already.
        if (spoken == 0 && parsed)
        {
            for (size_t i = 0; i < a.candidateGuids.size(); ++i)
            {
                bool alreadyTried = false;
                for (size_t s : speakers)
                    if (s == i)
                        alreadyTried = true;
                if (alreadyTried)
                    continue;

                Player* bot = ObjectAccessor::FindConnectedPlayer(ObjectGuid(a.candidateGuids[i]));
                if (!bot || !bot->IsInWorld() || !bot->IsAlive())
                    continue;

                Channel* channel = a.channelId ? OllamaResolveZoneChannel(bot, a.channelId) : nullptr;

                if (OllamaSubmitBotReply(bot, sender, a.msg, a.trimmedMsg, a.source, channel,
                                         a.chainDepth, a.scopeKey, a.senderIsBot))
                {
                    ++spoken;
                    spokeIdx.push_back(i);
                    firstSpeakerName = bot->GetName();
                    break;
                }
            }
        }

        if (g_DebugEnabled)
            LOG_INFO("module.ollamachat",
                     "[Ollama Chat] Addressee pass: {} candidates, {} named, parsed={}, {} speaking.",
                     a.candidateGuids.size(), named.size(), parsed, spoken);

        // Everyone who was in the running and did not end up speaking wanted to
        // answer and did not (plan 25 item 48).
        //
        // Only when the pass actually CHOSE. On the fallback path every
        // candidate speaks, so nobody held anything back -- and asking then
        // would spend a call to describe a silence that never happened.
        if (!g_HeldTongueEnable || !parsed || spoken == 0 || firstSpeakerName.empty() ||
            g_HeldTonguePrompt.empty())
            return;

        // Same generator as the room-pick above rather than the core's urand:
        // this file pulls in no core random header, and one chance roll is not
        // worth an include when <random> is already here.
        std::random_device heldRd;
        std::mt19937       heldGen(heldRd());
        std::uniform_int_distribution<uint32_t> heldRoll(0, 99);

        for (size_t i = 0; i < a.candidateGuids.size(); ++i)
        {
            bool spokeHere = false;
            for (size_t s : spokeIdx)
                if (s == i)
                    spokeHere = true;
            if (spokeHere)
                continue;

            if (heldRoll(heldGen) >= g_HeldTongueChance)
                continue;

            Player* quiet = ObjectAccessor::FindConnectedPlayer(ObjectGuid(a.candidateGuids[i]));
            if (!quiet || !quiet->IsInWorld() || !quiet->IsAlive())
                continue;

            OllamaHeldTongueRequest h;
            h.botGuid     = a.candidateGuids[i];
            h.botName     = quiet->GetName();
            h.fromName    = sender->GetName();
            h.message     = a.trimmedMsg;
            h.speakerName = firstSpeakerName;
            h.scopeKey    = a.scopeKey;
            h.prompt      = SafeFormat(g_HeldTonguePrompt,
                                       fmt::arg("bot_name", h.botName),
                                       fmt::arg("from_name", h.fromName),
                                       fmt::arg("message", h.message),
                                       fmt::arg("speaker_name", h.speakerName));

            if (h.prompt.empty() || h.prompt == "[Format Error]")
                continue;

            OllamaDispatch_SubmitHeldTongue(std::move(h));
        }
    }

    // By value: a repeated tail is trimmed off the line below, and the trimmed
    // text is what gets sent, recorded and echoed to the other bots.
    void Deliver(Completion c, const OllamaWorldSnapshot& world)
    {
        Player* bot = ObjectAccessor::FindConnectedPlayer(ObjectGuid(c.request.botGuid));
        if (!bot || !bot->IsInWorld())
            return;

        // The dead do not speak. This is the last mile -- every send in the
        // module goes through RouteMessage below -- so the check belongs here
        // even though it throws away work already paid for: a bot can die
        // while its line is in flight, which is exactly when a line about
        // dying was most likely generated. The witnesses who watched it fall
        // are alive, and they are the right voice for that anyway.
        if (!bot->IsAlive())
            return;

        PlayerbotAI* botAI = PlayerbotsMgr::instance().GetPlayerbotAI(bot);
        if (!botAI)
            return;

        const ObjectGuid botGuid = bot->GetGUID();

        // Direct address: the bot owes this person an answer, so it skips the
        // suppressions that exist to pace ambient chatter. Decided at submit
        // time in ProcessChat, where the Group was live to consult.
        const bool directAddress = c.request.directAddress;

        // Repetition is checked at delivery rather than at submission, because
        // we only know what the model actually said now.
        //
        // Skipped for direct address. Ask a bot the same question twice and
        // the same answer is correct -- suppressing it leaves the asker
        // staring at silence, which reads as the bot being broken rather than
        // as anti-repetition working.
        if (!directAddress && Governor_IsRepetitive(botGuid, c.request.scopeKey, c.text))
        {
            ++g_droppedGovernor;
            if (g_DebugEnabled)
                LOG_INFO("module.ollamachat",
                         "[Ollama Chat] Bot {} reply suppressed as repetitive: '{}'",
                         bot->GetName(), c.text);
            return;
        }

        // The whole answer is spared above, but an opener is not an answer.
        // Every line in a small party is direct address, so the check never ran
        // where it was needed most: ten of thirty-one measured lines opened
        // "Aye.", one bot eight times in nineteen. The history was recorded all
        // along -- only the looking was skipped.
        //
        // This does cost the occasional answer, and scope history records no
        // speaker, so it cannot tell another bot's opener from the bot's own.
        // OpenerCheckDirectAddress turns it off for anyone who would rather
        // hear a repeat than lose a reply.
        if (directAddress && g_OpenerCheckDirectAddress &&
            Governor_HasOpenerCollision(c.request.scopeKey, c.text))
        {
            ++g_droppedGovernor;
            if (g_DebugEnabled)
                LOG_INFO("module.ollamachat",
                         "[Ollama Chat] Bot {} reply suppressed for a repeated opener: '{}'",
                         bot->GetName(), c.text);
            return;
        }

        // A catchphrase riding along behind something new. Whole-line
        // suppression is skipped for direct address on purpose, and the opener
        // check only ever sees the first three words, so a repeated CLOSING
        // sentence passed both: measured 2026-09-17, one bot ended four
        // separate replies with "Keep your axe dry, cousin." and said "We walk
        // the bridge." three times.
        //
        // Trimmed rather than suppressed. The answer is still owed; only the
        // tic is not, and dropping the whole line would leave the asker staring
        // at silence -- the very thing direct address is exempted to prevent.
        if (directAddress && g_SentenceCheckDirectAddress)
        {
            std::string trimmed = Governor_StripRepeatedSentences(botGuid, c.text);
            if (trimmed != c.text)
            {
                c.text = std::move(trimmed);
                ++g_tailsTrimmed;

                if (g_DebugEnabled)
                    LOG_INFO("module.ollamachat",
                             "[Ollama Chat] Bot {} had a repeated sentence trimmed: '{}'",
                             bot->GetName(), c.text);
            }
        }

        if (!Governor_TryConsumeSend(botGuid, c.request.scopeKey, directAddress))
        {
            ++g_droppedGovernor;
            if (g_DebugEnabled)
                LOG_INFO("module.ollamachat",
                         "[Ollama Chat] Bot {} reply suppressed by cooldown/rate limit.",
                         bot->GetName());
            return;
        }

        Channel* channel = nullptr;
        if (!RouteMessage(bot, botAI, c, world, channel))
        {
            if (g_DebugEnabled)
                LOG_INFO("module.ollamachat",
                         "[Ollama Chat] Bot {} had nowhere to send its reply ({}).",
                         bot->GetName(), ChatChannelSourceLocalStr[c.request.source]);
            return;
        }

        Governor_RecordUtterance(botGuid, c.request.scopeKey, c.text);
        ++g_totalDelivered;

        // The line has landed, so anyone who held their tongue waiting for this
        // speaker can now be seen to have done so.
        NoteSpoken(c.request.scopeKey, bot->GetName());

        // This bot is now in a conversation with whoever it just answered, so
        // their next line in this scope is a turn in it rather than ambient
        // chatter. Only against a real person: an open conversation bypasses
        // pacing, and letting two bots open one with each other is how a
        // bot-to-bot loop would escape every brake in the governor.
        if (c.request.targetGuid)
        {
            Player* addressee = ObjectAccessor::FindConnectedPlayer(ObjectGuid(c.request.targetGuid));
            if (OllamaIsRealPlayer(addressee))
            {
                Governor_NoteConversation(botGuid, ObjectGuid(c.request.targetGuid),
                                          c.request.scopeKey);

                // And this bot now holds the thread here, so the person's next
                // line that names nobody stays with it instead of going to a
                // random voice (plan 25 item 54).
                Governor_NoteThreadHolder(botGuid, ObjectGuid(c.request.targetGuid),
                                          c.request.scopeKey);
            }
        }

        // Body language. Safe here and only here: this is the world thread.
        ScheduleBotExpression(bot, ObjectGuid(c.request.targetGuid), c.emoteId,
                              g_BotExpressionDelayMs);

        if (c.request.recordHistory && c.request.targetGuid)
        {
            AppendBotConversation(c.request.botGuid, c.request.targetGuid,
                                  c.request.originMessage, c.text);

            // Counts name mentions and, past a threshold, queues a
            // condensation or relationship revision. World thread.
            Player* target = ObjectAccessor::FindConnectedPlayer(ObjectGuid(c.request.targetGuid));
            Memory_NoteExchange(c.request.botGuid, c.request.targetGuid,
                                target ? target->GetName() : std::string(),
                                c.request.originMessage, c.text);
        }

        if (c.request.updateSentiment && c.request.targetGuid &&
            !c.request.originMessage.empty())
        {
            OllamaDispatch_SubmitSentiment(c.request.botGuid, c.request.targetGuid,
                                           c.request.originMessage);
        }

        if (g_DebugEnabled)
            LOG_INFO("module.ollamachat", "[Ollama Chat] {} ({}, depth {}): {}",
                     bot->GetName(), ChatChannelSourceLocalStr[c.request.source],
                     c.request.chainDepth, c.text);

        // Let other bots hear it -- with the chain depth advanced, which is
        // what stops the reply loop that had no brakes before.
        if (c.request.triggerBotReplies &&
            c.request.source != SRC_WHISPER_LOCAL)
        {
            ProcessBotChatMessage(bot, c.text, c.request.source, channel,
                                  static_cast<uint8_t>(c.request.chainDepth + 1));
        }
    }
}

// --------------------------------------------------------------------------

void OllamaDispatch_Start()
{
    std::lock_guard<std::mutex> lock(g_queueMutex);
    if (g_running)
        return;

    g_running = true;

    uint32_t count = g_DispatchWorkerThreads;
    if (count == 0)
        count = 4;
    if (count > 64)
        count = 64;

    g_workers.reserve(count);
    for (uint32_t i = 0; i < count; ++i)
        g_workers.emplace_back(WorkerLoop);

    LOG_INFO("module.ollamachat", "[Ollama Chat] Dispatcher started with {} worker threads.", count);
}

void OllamaDispatch_Stop()
{
    {
        std::lock_guard<std::mutex> lock(g_queueMutex);
        if (!g_running)
            return;
        g_running = false;
        g_queue.clear();
    }

    g_queueCv.notify_all();

    for (auto& worker : g_workers)
        if (worker.joinable())
            worker.join();

    g_workers.clear();

    {
        std::lock_guard<std::mutex> lock(g_doneMutex);
        g_done.clear();
    }

    LOG_INFO("module.ollamachat", "[Ollama Chat] Dispatcher stopped.");
}

bool OllamaDispatch_SubmitHeldTongue(OllamaHeldTongueRequest request)
{
    if (request.prompt.empty() || request.botGuid == 0)
        return false;

    Task task;
    task.type       = TaskType::HeldTongue;
    task.heldTongue = std::move(request);

    task.request.prompt  = task.heldTongue.prompt;
    task.request.kind    = OllamaRequestKind::Classify;
    task.request.botGuid = task.heldTongue.botGuid;
    task.request.botName = task.heldTongue.botName;

    {
        std::lock_guard<std::mutex> lock(g_queueMutex);

        if (!g_running)
            return false;

        // A stricter allowance than a reply gets: this is upkeep, and it must
        // never crowd out a line someone is actually waiting to hear.
        if (g_MaxQueueDepth > 0 && g_queue.size() >= g_MaxQueueDepth / 2)
        {
            ++g_droppedQueueFull;
            return false;
        }

        g_queue.push_back(std::move(task));
    }

    ++g_totalSubmitted;
    g_queueCv.notify_one();
    return true;
}

bool OllamaDispatch_SubmitAddressee(OllamaAddresseeRequest request)
{
    if (request.prompt.empty() || request.candidateGuids.empty())
        return false;

    Task task;
    task.type      = TaskType::Classify;
    task.addressee = std::move(request);

    // The worker reads the prompt and kind off task.request like every other
    // task; the addressee payload carries what the resolver needs afterwards.
    task.request.prompt  = task.addressee.prompt;
    task.request.kind    = OllamaRequestKind::Classify;
    task.request.botName = "addressee pass";

    {
        std::lock_guard<std::mutex> lock(g_queueMutex);

        if (!g_running)
            return false;

        if (g_MaxQueueDepth > 0 && g_queue.size() >= g_MaxQueueDepth)
        {
            ++g_droppedQueueFull;
            if (g_DebugEnabled)
                LOG_INFO("module.ollamachat",
                         "[Ollama Chat] Queue full ({}); dropping addressee pass.", g_queue.size());
            return false;
        }

        g_queue.push_back(std::move(task));
    }

    ++g_totalSubmitted;
    g_queueCv.notify_one();
    return true;
}

bool OllamaDispatch_Submit(OllamaChatRequest request)
{
    if (request.prompt.empty() || request.botGuid == 0)
        return false;

    Task task;
    task.type    = TaskType::ChatReply;
    task.request = std::move(request);

    {
        std::lock_guard<std::mutex> lock(g_queueMutex);

        if (!g_running)
            return false;

        if (g_MaxQueueDepth > 0 && g_queue.size() >= g_MaxQueueDepth)
        {
            ++g_droppedQueueFull;
            if (g_DebugEnabled)
                LOG_INFO("module.ollamachat",
                         "[Ollama Chat] Queue full ({}); dropping request for {}.",
                         g_queue.size(), task.request.botName);
            return false;
        }

        g_queue.push_back(std::move(task));
    }

    ++g_totalSubmitted;
    g_queueCv.notify_one();
    return true;
}

void OllamaDispatch_SubmitSentiment(uint64_t botGuid, uint64_t playerGuid,
                                    const std::string& message)
{
    if (!g_EnableSentimentTracking || message.empty())
        return;

    // Built here, on the world thread: the template is config state that
    // reload rewrites, so a worker must never read it.
    std::string prompt = BuildSentimentPrompt(message);
    if (prompt.empty())
        return;

    Task task;
    task.type                = TaskType::Sentiment;
    task.sentimentBotGuid    = botGuid;
    task.sentimentPlayerGuid = playerGuid;
    task.sentimentMessage    = message;
    task.sentimentPrompt     = std::move(prompt);

    {
        std::lock_guard<std::mutex> lock(g_queueMutex);
        if (!g_running)
            return;

        // Sentiment is best-effort; never let it crowd out actual chat.
        if (g_MaxQueueDepth > 0 && g_queue.size() >= g_MaxQueueDepth / 2)
            return;

        g_queue.push_back(std::move(task));
    }

    g_queueCv.notify_one();
}

void OllamaDispatch_Update(uint32_t /*diff*/)
{
    const auto now = Clock::now();

    // Move due completions out under the lock, then deliver without it: the
    // delivery path re-enters ProcessBotChatMessage, which submits new work.
    std::vector<Completion> due;

    {
        std::lock_guard<std::mutex> lock(g_doneMutex);

        for (auto it = g_done.begin(); it != g_done.end(); )
        {
            if (it->deliverAt <= now)
            {
                due.push_back(std::move(*it));
                it = g_done.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    // Deferred emotes whose line never came. Dropped rather than fired late:
    // "holds their tongue and lets X speak" asserts that X then spoke, and when
    // the reply was suppressed by the governor or had nowhere to go, X did not.
    // Saying it anyway describes a silence that was never broken.
    for (auto it = g_pendingEmotes.begin(); it != g_pendingEmotes.end(); )
    {
        if (it->giveUpAt <= now)
        {
            ++g_heldTongueAbandoned;
            it = g_pendingEmotes.erase(it);
        }
        else
        {
            ++it;
        }
    }

    // The speaker map is only ever consulted inside the wait window, so
    // anything older is dead weight. Pruned on size, so the common case of a
    // handful of entries costs nothing per tick.
    if (g_recentSpeakers.size() > 64)
    {
        for (auto it = g_recentSpeakers.begin(); it != g_recentSpeakers.end(); )
        {
            if (std::chrono::duration_cast<std::chrono::seconds>(now - it->second).count() >
                int64_t(g_HeldTongueEmoteWaitSeconds))
                it = g_recentSpeakers.erase(it);
            else
                ++it;
        }
    }

    if (due.empty())
        return;

    // One pass over the online player list for the whole tick, rather than one
    // per delivery.
    OllamaWorldSnapshot world;
    world.Build();

    for (const Completion& c : due)
    {
        try
        {
            // An addressee answer is not a line: it decides who speaks, and the
            // replies it submits come back through this same queue afterwards.
            if (c.isClassify)
                ResolveAddressee(c);
            else if (c.isHeldTongue)
                ResolveHeldTongue(c);
            else
                Deliver(c, world);
        }
        catch (const std::exception& e)
        {
            LOG_ERROR("module.ollamachat", "[Ollama Chat] Delivery exception: {}", e.what());
        }
    }
}

void OllamaChat_DispatchEmoteReaction(Player* bot, Player* player, uint32_t textEmote)
{
    if (!bot || !player)
        return;

    // This was the one producer in the module that never learned the party rule: it set SRC_SAY_LOCAL
    // unconditionally, so a bot answered a companion's gesture out loud to the room instead of to the
    // party (plan 38 §7). Ambient returns SRC_PARTY_LOCAL early and event chatter chooses on
    // partyAudience; this now makes the same choice they do.
    const bool partyAudience = bot->GetGroup() && !g_DisableForParty && OllamaGroupHasRealPlayer(bot);

    OllamaChatRequest request;
    request.botGuid    = bot->GetGUID().GetRawValue();
    request.targetGuid = player->GetGUID().GetRawValue();
    request.source     = partyAudience ? SRC_PARTY_LOCAL : SRC_SAY_LOCAL;
    request.chainDepth = 0;
    request.botName    = bot->GetName();
    request.kind       = OllamaRequestKind::EventChatter;
    // Party lines key on the GROUP, exactly as they do in ProcessChat and in event chatter -- keying
    // them on the zone puts the reply in a different conversation space from the party chat it was
    // said in, so it counts for no cooldown, no repetition history and no thread.
    request.scopeKey   = partyAudience && bot->GetGroup()
                             ? Governor_MakeScopeKey("Party", 0, "", 0, bot->GetGroup()->GetGUID().GetCounter())
                             : Governor_MakeScopeKey("Say", 0, "", 0, bot->GetZoneId());
    request.triggerBotReplies = false;
    // Someone emoted at this bot by name. Its own debounce paces this;
    // the ambient say cooldown has no business also silencing it.
    request.directAddress = true;
    // The reply is addressed to a person, so it belongs in the history that condensation reads.
    // Without this the whole emote path was invisible to memory (plan 38 §2.2).
    request.recordHistory = true;

    uint32_t maxWords = 0;
    request.prompt = BuildEmoteReactionPrompt(bot, player, textEmote, &maxWords);
    if (request.prompt.empty())
        return;

    request.maxWords = maxWords;

    OllamaDispatch_Submit(std::move(request));
}

namespace
{
    // Background upkeep must never crowd out actual chat, so it gets a
    // stricter queue allowance than a reply does.
    bool SubmitBackground(Task&& task)
    {
        std::lock_guard<std::mutex> lock(g_queueMutex);
        if (!g_running)
            return false;
        if (g_MaxQueueDepth > 0 && g_queue.size() >= g_MaxQueueDepth / 2)
            return false;

        g_queue.push_back(std::move(task));
        return true;
    }
}

bool OllamaDispatch_SubmitCondensation(uint64_t botGuid, const std::string& prompt)
{
    if (botGuid == 0 || prompt.empty())
        return false;

    Task task;
    task.type          = TaskType::Condense;
    task.memoryBotGuid = botGuid;
    task.memoryPrompt  = prompt;

    // Report the refusal (plan 41 M5). The caller sets a flag before asking and
    // clears it when the work completes; a silently dropped submit left that
    // flag set forever.
    if (!SubmitBackground(std::move(task)))
        return false;

    g_queueCv.notify_one();
    return true;
}

bool OllamaDispatch_SubmitEventDigest(uint64_t botGuid, const std::string& prompt)
{
    if (botGuid == 0 || prompt.empty())
        return false;

    Task task;
    task.type          = TaskType::EventDigest;
    task.memoryBotGuid = botGuid;
    task.memoryPrompt  = prompt;

    // Same contract as the condensation submit (plan 41 M5): a refusal has to
    // reach the caller, or eventFlushing stays true and that bot never digests
    // another deed for the life of the process.
    if (!SubmitBackground(std::move(task)))
        return false;

    g_queueCv.notify_one();
    return true;
}

void OllamaDispatch_SubmitRelationship(uint64_t botGuid, uint64_t otherGuid,
                                       const std::string& otherName,
                                       const std::string& prompt)
{
    if (botGuid == 0 || otherGuid == 0 || prompt.empty())
        return;

    Task task;
    task.type            = TaskType::Relationship;
    task.memoryBotGuid   = botGuid;
    task.memoryOtherGuid = otherGuid;
    task.memoryOtherName = otherName;
    task.memoryPrompt    = prompt;

    if (SubmitBackground(std::move(task)))
        g_queueCv.notify_one();
}

OllamaDispatchStats OllamaDispatch_GetStats()
{
    OllamaDispatchStats stats{};

    {
        std::lock_guard<std::mutex> lock(g_queueMutex);
        stats.queuedRequests = static_cast<uint32_t>(g_queue.size());
        stats.workers        = static_cast<uint32_t>(g_workers.size());
    }
    {
        std::lock_guard<std::mutex> lock(g_doneMutex);
        stats.pendingDeliveries = static_cast<uint32_t>(g_done.size());
    }
    {
        std::lock_guard<std::mutex> lock(g_errorMutex);
        stats.lastError = g_lastError;
    }

    stats.inFlight              = g_inFlight.load();
    stats.totalSubmitted        = g_totalSubmitted.load();
    stats.totalDelivered        = g_totalDelivered.load();
    stats.totalDroppedQueueFull = g_droppedQueueFull.load();
    stats.totalDroppedEmpty     = g_droppedEmpty.load();
    stats.totalDroppedGovernor  = g_droppedGovernor.load();
    stats.totalFailed           = g_totalFailed.load();
    stats.heldTongueDeferred    = g_heldTongueDeferred.load();
    stats.heldTongueFired       = g_heldTongueFired.load();
    stats.heldTongueAbandoned   = g_heldTongueAbandoned.load();
    stats.tailsTrimmed          = g_tailsTrimmed.load();

    return stats;
}

Channel* OllamaResolveZoneChannel(Player* bot, uint32_t chatChannelId)
{
    if (!bot)
        return nullptr;

    ChannelMgr* mgr = ChannelMgr::forTeam(bot->GetTeamId());
    if (!mgr)
        return nullptr;

    PlayerbotAI* botAI = PlayerbotsMgr::instance().GetPlayerbotAI(bot);
    std::string zoneName;
    if (botAI)
        if (AreaTableEntry const* zone = botAI->GetCurrentZone())
            zoneName = PlayerbotAI::GetLocalizedAreaName(zone);

    for (auto const& [key, channel] : mgr->GetChannels())
    {
        if (!channel || channel->GetName().empty())
            continue;
        if (channel->GetChannelId() != chatChannelId)
            continue;

        // Global channels (LFG, WorldDefense) are not zone-scoped.
        const bool zoneScoped = (chatChannelId != uint32_t(ChatChannelId::LOOKING_FOR_GROUP) &&
                                 chatChannelId != uint32_t(ChatChannelId::WORLD_DEFENSE));

        if (zoneScoped)
        {
            if (zoneName.empty())
                continue;
            if (channel->GetName().find(zoneName) == std::string::npos)
                continue;
        }

        return channel;
    }

    return nullptr;
}
