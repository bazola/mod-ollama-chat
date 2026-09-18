#ifndef MOD_OLLAMA_CHAT_GOVERNOR_H
#define MOD_OLLAMA_CHAT_GOVERNOR_H

#include "ObjectGuid.h"
#include <string>
#include <cstdint>

// --------------------------------------------------------------------------
// Conversation governor.
//
// Four independent brakes, because bot chat fails in four different ways:
//
//   1. Chain depth + decay  -- stops a bot->bot reply chain running away.
//   2. The audience rule    -- bots only answer other bots while a real player
//                              has spoken in that scope recently.
//   3. Cooldowns and rates  -- stops one bot, or one crowd, dominating.
//   4. Repetition scoring   -- stops the same line, and the same opener,
//                              being said twice.
//
// All entry points are mutex-guarded; the dispatcher consults some of these
// from worker threads.
// --------------------------------------------------------------------------

// Build the key that identifies a conversation space. Channel messages key on
// the channel name, guild chat on the guild id, party/raid on the group, and
// say/yell on the zone so that proximity chat in one zone does not rate-limit
// another.
std::string Governor_MakeScopeKey(const char* sourceName, uint32_t channelId,
                                  const std::string& channelName,
                                  uint32_t guildId, uint32_t groupOrZoneId);

// --- the audience rule ----------------------------------------------------

void Governor_NoteHumanMessage(const std::string& scopeKey);
bool Governor_HasRecentHuman(const std::string& scopeKey);

// --- chain depth ----------------------------------------------------------

bool     Governor_ChainDepthAllowed(uint8_t depth);
uint32_t Governor_ApplyChainDecay(uint32_t baseChancePct, uint8_t depth);

// --- open conversations ---------------------------------------------------

// A bot that has already answered someone is in a conversation with them, and
// the next thing that person says is a turn in it rather than ambient chatter.
// This is what makes a bot keep talking after the opening exchange without
// having to be named again every single line.
//
// Recorded per scope, so answering someone in party does not exempt this bot
// from pacing in General. Only ever recorded against a real player.
void Governor_NoteConversation(ObjectGuid botGuid, ObjectGuid playerGuid,
                               const std::string& scopeKey);
bool Governor_InConversation(ObjectGuid botGuid, ObjectGuid playerGuid,
                             const std::string& scopeKey);

// --- the thread holder ----------------------------------------------------

// Who a person is actually mid-exchange with here (plan 25 item 54).
//
// The open-conversation map above cannot serve this. It answers "is THIS bot
// talking to them", per bot -- and routing has to ask the other question,
// "which bot holds the thread here", before it knows which bot to ask about.
// That is why the state existed and routing still picked at random: measured
// 2026-09-17, 11 of 21 unnamed follow-ups went to a bot that was not the one
// the person had been talking to.
//
// Set when a bot's line aimed at a real person is delivered, and when a person
// names a bot. Never recorded against another bot: a thread is something a
// person is in.
void Governor_NoteThreadHolder(ObjectGuid botGuid, ObjectGuid playerGuid,
                               const std::string& scopeKey);

// The bot holding the thread with this person here, or 0 when none is live.
// Never returns a stale holder -- the window is checked on read, and it widens
// a little as an exchange runs on, because a conversation four turns deep
// survives a longer pause than one that has only just started.
uint64_t Governor_ThreadHolder(ObjectGuid playerGuid, const std::string& scopeKey);

// --- cooldowns and rate limits -------------------------------------------

// Checks per-bot cooldown, per-scope cooldown, scope rate and global rate.
// On success the send is reserved (all counters advance), so call this exactly
// once per message actually being committed.
// `directAddress` marks a reply the bot owes someone who spoke straight to it
// -- a whisper. Those skip the per-bot and per-scope pacing cooldowns, which
// exist to stop ambient chatter running hot and have no business silencing an
// answer to a direct question. The global messages-per-minute ceiling still
// applies: that one protects the LLM backend, not the pacing.
bool Governor_TryConsumeSend(ObjectGuid botGuid, const std::string& scopeKey,
                             bool directAddress = false);

// Same checks without reserving, for deciding whether to spend an LLM call.
bool Governor_CanSend(ObjectGuid botGuid, const std::string& scopeKey,
                      bool directAddress = false);

// --- repetition -----------------------------------------------------------

// True when text is too close to something this bot said recently, or to
// something recently said in this scope, or reuses a recent opener.
bool Governor_IsRepetitive(ObjectGuid botGuid, const std::string& scopeKey,
                           const std::string& text);

// True when this scope heard the same opener recently. The opener half of
// Governor_IsRepetitive on its own, so a path that must not suppress a whole
// answer can still refuse a line that starts like the last one. Scope history
// records no speaker, so this cannot tell "another bot said it" from "this bot
// is repeating itself" -- it answers only "was this opener just used here".
bool Governor_HasOpenerCollision(const std::string& scopeKey, const std::string& text);

void Governor_RecordUtterance(ObjectGuid botGuid, const std::string& scopeKey,
                              const std::string& text);

// Drop sentences this bot has said recently from a line it is about to say,
// keeping the rest of the answer.
//
// Whole-line suppression is skipped for direct address on purpose, and in a
// small party every line is direct address -- so a bot's favourite closing
// sentence is checked by nothing at all. The opener check does not reach it
// either: a tail is not an opener. Measured in the 2026-09-17 playtest: one bot
// ended four separate replies with "Keep your axe dry, cousin." and said "We
// walk the bridge." three times; another repeated a two-sentence refusal
// verbatim thirteen minutes apart.
//
// Only this bot's own history is consulted. Two people can reach the same
// sentence honestly; one person reaching it twice is the tic.
//
// Returns the line with repeated sentences removed. If every sentence is a
// repeat the ORIGINAL comes back untouched -- the same question deserves the
// same answer, and an empty line reads as the bot being broken.
std::string Governor_StripRepeatedSentences(ObjectGuid botGuid, const std::string& text);

// --- what was just said here ---------------------------------------------

// A line and who said it, kept per scope. The repetition history above cannot
// serve this: it stores normalized text and records no speaker at all.
void Governor_NoteScopeLine(const std::string& scopeKey, const std::string& speakerName,
                            const std::string& text);

// The recent lines of this scope as "Name: line", oldest first, for a prompt
// that has to work out who a follow-up was aimed at. `skipMostRecent` drops
// that many from the end -- one, normally, because the line being decided about
// has already been recorded by the time the decision is made.
std::string Governor_RecentLines(const std::string& scopeKey, uint32_t maxLines,
                                 uint32_t skipMostRecent);

// Similarity in [0,1]; exposed for the .ollama status command and testing.
float Governor_Similarity(const std::string& a, const std::string& b);

// --- per-feature debounces -----------------------------------------------

bool Governor_TryConsumeEmoteReaction(ObjectGuid botGuid, ObjectGuid playerGuid);

// Replaces the old raw-Player*-keyed event cooldown map. Only consumes the
// cooldown when the event is actually committed.
bool Governor_TryConsumeEventCooldown(ObjectGuid botGuid);

// --- maintenance ----------------------------------------------------------

void Governor_OnPlayerLogout(ObjectGuid guid);
void Governor_Update();   // prune expired state; called from the world tick
void Governor_Reset();

// Snapshot for the .ollama status command.
struct GovernorStats
{
    uint32_t trackedBots;
    uint32_t trackedScopes;
    uint32_t sendsLastMinute;
    uint32_t blockedCooldown;
    uint32_t blockedRate;
    uint32_t blockedRepetition;
    uint32_t blockedChainDepth;
    uint32_t blockedNoAudience;
};
GovernorStats Governor_GetStats();

#endif // MOD_OLLAMA_CHAT_GOVERNOR_H
