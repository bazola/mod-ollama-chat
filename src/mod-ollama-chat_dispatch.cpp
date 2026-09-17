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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <list>
#include <mutex>
#include <thread>
#include <vector>

// Local patch (custom-wow): mod-ledger records channel replies, which bypass the
// OnPlayerCanUseChat hooks. Weak so this links without it.
void LedgerRecordBotChat(Player* bot, uint32 type, std::string const& msg, Channel* channel) __attribute__((weak));

namespace
{
    using Clock = std::chrono::steady_clock;

    enum class TaskType : uint8_t { ChatReply, Sentiment, Condense, Relationship, Classify };

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

            case SRC_PARTY_LOCAL:
                if (g_DisableForParty || !bot->GetGroup())
                    return false;
                if (!OllamaGroupHasRealPlayer(bot))
                    return false;
                return botAI->SayToParty(c.text);

            case SRC_RAID_LOCAL:
                if (g_DisableForParty || !bot->GetGroup())
                    return false;
                if (!OllamaGroupHasRealPlayer(bot))
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
        bool parsed = false;

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
            }
            catch (const std::exception&)
            {
                parsed = false;
            }
        }

        std::vector<size_t> speakers;

        auto takeAll = [&]()
        {
            speakers.clear();
            for (size_t i = 0; i < a.candidateGuids.size(); ++i)
                speakers.push_back(i);
        };

        if (!parsed)
        {
            takeAll();
        }
        else if (named.empty())
        {
            // Said to the room rather than to anyone. One voice answers instead
            // of all of them, which is the point of the pass.
            if (!a.candidateGuids.empty())
            {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<size_t> pick(0, a.candidateGuids.size() - 1);
                speakers.push_back(pick(gen));
            }
        }
        else
        {
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

        uint32_t spoken = 0;
        for (size_t idx : speakers)
        {
            if (idx >= a.candidateGuids.size())
                continue;

            Player* bot = ObjectAccessor::FindConnectedPlayer(ObjectGuid(a.candidateGuids[idx]));
            if (!bot || !bot->IsInWorld() || !bot->IsAlive())
                continue;

            Channel* channel = a.channelId ? OllamaResolveZoneChannel(bot, a.channelId) : nullptr;

            if (OllamaSubmitBotReply(bot, sender, a.msg, a.trimmedMsg, a.source, channel,
                                     a.chainDepth, a.scopeKey, a.senderIsBot))
            {
                ++spoken;
                if (parsed && a.maxSpeakers > 0 && spoken >= a.maxSpeakers)
                    break;
            }
        }

        if (g_DebugEnabled)
            LOG_INFO("module.ollamachat",
                     "[Ollama Chat] Addressee pass: {} candidates, {} named, parsed={}, {} speaking.",
                     a.candidateGuids.size(), named.size(), parsed, spoken);
    }

    void Deliver(const Completion& c, const OllamaWorldSnapshot& world)
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

        // This bot is now in a conversation with whoever it just answered, so
        // their next line in this scope is a turn in it rather than ambient
        // chatter. Only against a real person: an open conversation bypasses
        // pacing, and letting two bots open one with each other is how a
        // bot-to-bot loop would escape every brake in the governor.
        if (c.request.targetGuid)
        {
            Player* addressee = ObjectAccessor::FindConnectedPlayer(ObjectGuid(c.request.targetGuid));
            if (OllamaIsRealPlayer(addressee))
                Governor_NoteConversation(botGuid, ObjectGuid(c.request.targetGuid),
                                          c.request.scopeKey);
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

    OllamaChatRequest request;
    request.botGuid    = bot->GetGUID().GetRawValue();
    request.targetGuid = player->GetGUID().GetRawValue();
    request.source     = SRC_SAY_LOCAL;
    request.chainDepth = 0;
    request.botName    = bot->GetName();
    request.kind       = OllamaRequestKind::EventChatter;
    request.scopeKey   = Governor_MakeScopeKey("Say", 0, "", 0, bot->GetZoneId());
    request.triggerBotReplies = false;
    // Someone emoted at this bot by name. Its own debounce paces this;
    // the ambient say cooldown has no business also silencing it.
    request.directAddress = true;

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

void OllamaDispatch_SubmitCondensation(uint64_t botGuid, const std::string& prompt)
{
    if (botGuid == 0 || prompt.empty())
        return;

    Task task;
    task.type          = TaskType::Condense;
    task.memoryBotGuid = botGuid;
    task.memoryPrompt  = prompt;

    if (SubmitBackground(std::move(task)))
        g_queueCv.notify_one();
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
