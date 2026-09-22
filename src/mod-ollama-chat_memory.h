#ifndef MOD_OLLAMA_CHAT_MEMORY_H
#define MOD_OLLAMA_CHAT_MEMORY_H

#include "ObjectGuid.h"
#include <string>
#include <unordered_set>
#include <vector>
#include <cstdint>

class Player;

// --------------------------------------------------------------------------
// Long-term memory and relationships.
//
// Conversation history alone is a sliding window: once a line falls out of it,
// the bot has no idea it ever happened. That is why bots feel like they meet
// you fresh every session no matter how much you have talked.
//
// Two mechanisms fix that, both bounded so the prompt never grows without
// limit:
//
//   MEMORY        History accumulates until it crosses a token budget. At that
//                 point it is condensed by the model into a handful of short
//                 narrator-style memories, each with an importance score, and
//                 the raw history is cleared. At prompt-build time the most
//                 important memories are selected within a separate, smaller
//                 token budget.
//
//   RELATIONSHIPS When a name comes up often enough in a bot's history, the
//                 model is asked to write (or revise) a sentence on how that
//                 bot feels about that person. That sentence goes into future
//                 prompts, so a bot's attitude persists and evolves instead of
//                 resetting.
//
// Both work in normal mode and roleplay mode; roleplay mode simply asks for
// in-character phrasing and allows a larger budget.
//
// Sentiment tracking (a single number) is complementary and unchanged: this is
// the qualitative half.
// --------------------------------------------------------------------------

struct BotMemoryEntry
{
    std::string text;
    uint8_t     importance = 5;   // 1..10
    uint64_t    createdAt  = 0;   // unix seconds
};

struct BotRelationship
{
    uint64_t    otherGuid = 0;
    std::string otherName;
    std::string description;
    uint32_t    mentions  = 0;
    uint64_t    updatedAt = 0;
};

void Memory_Load();
void Memory_LoadHouseholds();
bool Memory_MayTell(const std::string& text, const std::unordered_set<uint32_t>& presentAccounts);
void Memory_SaveAll();
void Memory_ForgetBot(ObjectGuid botGuid);

// Called on the world thread after a bot exchange is recorded. Counts name
// mentions and, when a threshold is crossed, queues a condensation or a
// relationship update through the dispatcher.
void Memory_NoteExchange(uint64_t botGuid, uint64_t otherGuid,
                         const std::string& otherName,
                         const std::string& incomingMessage,
                         const std::string& botReply);

// Record one thing a bot will remember, without going through condensation.
//
// Condensation is the only writer this store has ever had, and it produces
// memories in batches from accumulated history. A held tongue (plan 25 item 48)
// is a single fact that exists at one moment and would be gone by the time
// history was distilled, so it needs a direct way in.
//
// Importance is the model's own weight for the thought, 1..10, which is what
// Memory_BuildPromptSection sorts on -- so a thought that mattered outranks
// idle ones inside the prompt's token budget.
//
// Trims to Memory.MaxPerBot itself, dropping the least important: the cap is
// otherwise only applied on the condensation path, which may never run.
//
// Thread-safe. Marks the bot dirty so the periodic save persists it.
void Memory_Remember(uint64_t botGuid, const std::string& text, uint8_t importance);

// Buffer one notable thing a bot did or watched happen (plan 38).
//
// Condensation can only ever distil what was SAID to a bot and answered, so a
// night of questing, dying and killing leaves no trace at all: a bot can stand
// over a dead boss and have no way to remember it. This is the other door in.
// Events accumulate per bot and are digested by the model in one call, so the
// cost is one request per handful of deeds rather than one per kill.
//
// `line` is already plain third-person English naming the actor, the deed and
// the place -- built on the world thread, because the place and the company can
// only be read there.
//
// World thread only. Submits the digest itself once the buffer is full.
void Memory_NoteGameEvent(uint64_t botGuid, const std::string& line);

// Flush any part-filled event buffer that has gone stale, so a bot that saw
// three things and then walked away still writes them down. World thread only;
// called from the module's maintenance tick.
void Memory_FlushStaleEvents();

// Who keeps deed memories at all (plan 38).
//
// Fleet-wide, every bot digesting its own deeds cost ~20 model calls a minute
// and filled the store with weather: 892 memories in thirteen minutes, 77% of
// which named no place. Only a bot that has stood in a company with a real
// person keeps them -- and once marked it keeps them for the nights it spends
// out alone too, which is the whole value of having a companion with a life.
//
// Load at startup; note one the moment it is seen grouped with a person.
void Memory_LoadCompanions();
void Memory_NoteCompanion(uint64_t botGuid);
bool Memory_IsCompanion(uint64_t botGuid);

// Prompt fragments. World thread only.
//
// `about` may be null; when set, that person's relationship line is listed
// first so the bot's attitude to whoever it is talking to always survives the
// budget.
std::string Memory_BuildPromptSection(Player* bot, Player* about);

// --- worker-side entry points (called from the dispatcher) ----------------

// Condense a bot's accumulated history into memories, then clear the history.
void Memory_RunCondensation(uint64_t botGuid, const std::string& prompt);

// Digest a bot's buffered deeds into memories, then clear the buffer.
void Memory_RunEventDigest(uint64_t botGuid, const std::string& prompt);

// Write or revise how a bot feels about someone.
void Memory_RunRelationshipUpdate(uint64_t botGuid, uint64_t otherGuid,
                                  const std::string& otherName,
                                  const std::string& prompt);

// --- prompt builders (world thread; they read config templates) -----------

std::string Memory_BuildCondensationPrompt(Player* bot);
std::string Memory_BuildEventPrompt(Player* bot);
std::string Memory_BuildRelationshipPrompt(Player* bot, uint64_t otherGuid,
                                           const std::string& otherName);

// Rough token estimate. Deliberately cheap: characters / 4, the usual
// approximation, which is accurate enough for a budget.
uint32_t Memory_EstimateTokens(const std::string& text);

#endif // MOD_OLLAMA_CHAT_MEMORY_H
