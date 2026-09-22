#ifndef MOD_OLLAMA_CHAT_DISPATCH_H
#define MOD_OLLAMA_CHAT_DISPATCH_H

#include "mod-ollama-chat_capability.h"
#include "mod-ollama-chat_handler.h"

#include "ObjectGuid.h"
#include <string>
#include <cstdint>
#include <vector>

class Player;
class Channel;

// --------------------------------------------------------------------------
// Central dispatcher.
//
// THE RULE THIS FILE EXISTS TO ENFORCE:
//   worker threads do HTTP and string work only.
//   every read or write of a Player, Channel, Guild, Group or Map happens on
//   the world thread.
//
// Previously each candidate bot spawned its own detached std::thread which
// then called ObjectAccessor, Channel::Say, botAI->Say and re-entered the
// whole eligibility scan -- all racing Map::Update, and all unbounded.
//
// Now: the caller builds the prompt on the world thread and submits. A fixed
// pool of workers performs the generation. Completions come back through a
// queue that OllamaDispatch_Update() drains on the world thread, which is the
// only place delivery happens.
// --------------------------------------------------------------------------

// Resolve the zone-scoped instance of a numbered channel (General, Trade, ...)
// for this bot.
//
// ChannelMgr keys channels by their FULL name, and zone channels are named
// "General - Elwynn Forest", not "General" -- so GetChannel("General") always
// returns nullptr. Matching is by channel id plus zone-name containment, the
// same way PlayerbotAI::SayToChannel does it.
//
// World thread only.
Channel* OllamaResolveZoneChannel(Player* bot, uint32_t chatChannelId);

struct OllamaChatRequest
{
    // Who speaks, and to whom.
    uint64_t botGuid    = 0;
    uint64_t targetGuid = 0;      // 0 when there is no specific addressee

    // Where the line goes.
    ChatChannelSourceLocal source = SRC_SAY_LOCAL;
    std::string            channelName;
    uint32_t               channelId = 0;

    // Loop control. Human-originated messages start at depth 0; each bot reply
    // to a bot increments it.
    uint8_t     chainDepth = 0;
    std::string scopeKey;

    // This line was aimed at this bot in particular -- whispered, named, or
    // said in a small party it belongs to -- rather than said to the room.
    // Decided on the world thread at submit time, where the Group is live, and
    // carried here so delivery does not have to guess it back from `source`.
    // Direct address skips the pacing cooldowns and repetition suppression.
    bool directAddress = false;

    // Generation.
    std::string       prompt;
    OllamaRequestKind kind = OllamaRequestKind::ChatReply;

    // Resolved on the world thread before submission so the worker never has
    // to touch a Player to know these.
    std::string botName;
    // Most words the spoken line may keep, from the "@N" on the length instruction drawn for it; 0 = no cap.
    uint32_t    maxWords = 0;
    std::string originMessage;    // the message being replied to, if any

    // Held back this much longer before the line is delivered, on top of the
    // typing simulation. The group branch staggers its speakers with it: two
    // replies landing in the same tick read as a pile-on even when the pass
    // chose both of them on purpose.
    uint32_t    extraDelayMs = 0;

    // Post-delivery behaviour.
    bool triggerBotReplies = true;   // let other bots hear this line
    bool recordHistory     = false;  // append to conversation history
    bool updateSentiment   = false;  // run sentiment analysis on originMessage
};

// Submit a request. Returns false when the queue is at MaxQueueDepth, in which
// case the caller should simply skip this bot rather than pile up backlog.
bool OllamaDispatch_Submit(OllamaChatRequest request);

// The addressee pass: one cheap call that decides who a line was aimed at,
// before anyone spends a generation answering it.
//
// Submitted from the world thread with the bots that passed their rolls,
// answered on a worker by the cheap lane, and resolved back on the world
// thread -- which then submits the real replies for whoever was addressed.
// Everything needed to replay that submission travels in here, because by the
// time the answer lands, ProcessChat's locals are long gone.
//
// Degrades to the old behaviour on any failure: a lane that is down, a reply
// that will not parse, or a name that matches nobody all fall back to letting
// every candidate answer, which is exactly what would have happened without
// the pass.
struct OllamaAddresseeRequest
{
    uint64_t               senderGuid = 0;
    std::string            msg;
    std::string            trimmedMsg;
    ChatChannelSourceLocal source = SRC_SAY_LOCAL;
    uint32_t               channelId = 0;
    uint8_t                chainDepth = 0;
    std::string            scopeKey;
    bool                   senderIsBot = false;

    std::vector<uint64_t>    candidateGuids;
    std::vector<std::string> candidateNames;   // parallel to candidateGuids
    uint32_t                 maxSpeakers = 1;

    // Who this person was already mid-exchange with when the line was routed,
    // snapshotted on the world thread at submit time rather than read back
    // afterwards: the resolver runs a round trip later, and the question is who
    // held the thread when they spoke. 0 when nobody did.
    uint64_t    holderGuid = 0;
    std::string holderName;

    std::string prompt;
};

bool OllamaDispatch_SubmitAddressee(OllamaAddresseeRequest request);

// A held tongue (plan 25 item 48). When the addressee pass picks one speaker,
// the candidates it passed over wanted to answer and did not. This asks the
// cheap lane what one of them kept back and how much it mattered, in a single
// JSON answer, then either surfaces it as an emote or only remembers it.
//
// Same two-stage shape as the addressee pass: submitted from the world thread,
// answered on a worker, resolved back on the world thread -- which is where
// both the emote and the memory write have to happen.
//
// Nothing here is ever spoken, and the withheld text is never displayed, only
// the fact of it. A lane that is down or an answer that will not parse means no
// thought, no emote, and the bot is simply silent as it is today.
struct OllamaHeldTongueRequest
{
    uint64_t    botGuid = 0;
    std::string botName;

    std::string fromName;      // who said the line that went unanswered
    std::string message;       // what they said
    std::string speakerName;   // who answered instead

    // Where this happened. Carried so the emote can wait for the line it
    // defers to: the resolver has to recognise that speaker's reply landing in
    // this conversation, not somewhere else the same bot happens to be talking.
    std::string scopeKey;

    std::string prompt;
};

bool OllamaDispatch_SubmitHeldTongue(OllamaHeldTongueRequest request);

// Drain finished generations and deliver them. World thread only.
void OllamaDispatch_Update(uint32_t diff);

void OllamaDispatch_Start();
void OllamaDispatch_Stop();

// Fire-and-forget sentiment analysis. Runs entirely on a worker: the sentiment
// store is mutex-guarded in-memory state plus async DB writes, so it never
// needs the world thread -- and it must never run on it, because the analysis
// is a blocking HTTP call.
void OllamaDispatch_SubmitSentiment(uint64_t botGuid, uint64_t playerGuid,
                                    const std::string& message);

// Distil a bot's accumulated history into lasting memories, then clear it.
// Fire-and-forget; runs entirely on a worker.
// Return false when the queue refused the work (full, or shutting down), so the
// caller can release the claim flag it set before asking (plan 41 M5).
bool OllamaDispatch_SubmitCondensation(uint64_t botGuid, const std::string& prompt);

// Distil a bot's buffered deeds into lasting memories (plan 38). Same shape as
// condensation, and for the same reason: it ends in a mutex-guarded store
// write, never in a spoken line, so it needs no trip back to the world thread.
bool OllamaDispatch_SubmitEventDigest(uint64_t botGuid, const std::string& prompt);

// Write or revise how a bot feels about someone.
void OllamaDispatch_SubmitRelationship(uint64_t botGuid, uint64_t otherGuid,
                                       const std::string& otherName,
                                       const std::string& prompt);

// A player emoted at a bot and the bot decided to answer in words.
// World thread only.
void OllamaChat_DispatchEmoteReaction(Player* bot, Player* player, uint32_t textEmote);

struct OllamaDispatchStats
{
    uint32_t queuedRequests;
    uint32_t inFlight;
    uint32_t pendingDeliveries;
    uint32_t workers;
    uint64_t totalSubmitted;
    uint64_t totalDelivered;
    uint64_t totalDroppedQueueFull;
    uint64_t totalDroppedEmpty;
    uint64_t totalDroppedGovernor;
    uint64_t totalFailed;

    // Held-tongue emotes: held back waiting for the line they defer to, sent
    // once it landed, and given up on because it never did.
    uint64_t heldTongueDeferred;
    uint64_t heldTongueFired;
    uint64_t heldTongueAbandoned;

    // Replies that went out with a repeated sentence trimmed off.
    uint64_t tailsTrimmed;

    std::string lastError;
};
OllamaDispatchStats OllamaDispatch_GetStats();

#endif // MOD_OLLAMA_CHAT_DISPATCH_H
