#ifndef MOD_OLLAMA_CHAT_SENTIMENT_H
#define MOD_OLLAMA_CHAT_SENTIMENT_H

#include <string>
#include <cstdint>
#include "Player.h"

// --------------------------------------------
// Sentiment Tracking Functions
// --------------------------------------------

/**
 * Get the current sentiment value between a bot and player
 * @param botGuid GUID of the bot
 * @param playerGuid GUID of the player
 * @return Sentiment value (0.0-1.0), or default value if not found
 */
float GetBotPlayerSentiment(uint64_t botGuid, uint64_t playerGuid);

/**
 * Set the sentiment value between a bot and player
 * @param botGuid GUID of the bot
 * @param playerGuid GUID of the player
 * @param sentimentValue New sentiment value (0.0-1.0)
 */
void SetBotPlayerSentiment(uint64_t botGuid, uint64_t playerGuid, float sentimentValue);

/**
 * Analyze the sentiment of a message using LLM
 * @param message The message to analyze
 * @return Sentiment adjustment (-1.0 to 1.0)
 */
float AnalyzeMessageSentiment(const std::string& prompt);

/**
 * Run the sentiment analysis LLM call and store the result.
 *
 * BLOCKING -- this performs an HTTP request. Call it from a dispatcher worker,
 * never from the world thread. It touches only the mutex-guarded sentiment map
 * and async DB writes, so it needs no world-thread access of its own.
 */
void ApplySentimentAnalysis(uint64_t botGuid, uint64_t playerGuid,
                            const std::string& message, const std::string& prompt);

// Format the sentiment prompt. World thread only -- it reads the config
// template, which reload rewrites.
std::string BuildSentimentPrompt(const std::string& message);

/**
 * Queue a sentiment update for a player's message to a bot.
 *
 * Non-blocking: hands the work to the dispatcher. The old version called
 * AnalyzeMessageSentiment() inline, which would stall whichever thread it ran
 * on for a full LLM round trip.
 */
void UpdateBotPlayerSentiment(Player* bot, Player* player, const std::string& message);

/**
 * Get sentiment prompt addition for including in bot responses
 * @param bot The bot
 * @param player The player they're responding to
 * @return Formatted sentiment prompt addition
 */
std::string GetSentimentPromptAddition(Player* bot, Player* player);

/**
 * Load all sentiment data from database into memory
 */
void LoadBotPlayerSentimentsFromDB();

/**
 * Save all sentiment data from memory to database
 */
void SaveBotPlayerSentimentsToDB();

/**
 * Initialize the sentiment tracking system
 */
void InitializeSentimentTracking();

// --------------------------------------------
// Regard (local patch, plan 14)
// --------------------------------------------

// World tick: reloads the regard table on a background thread every
// OllamaChat.Regard.RefreshSeconds. Cheap when disabled.
void Regard_Tick(uint32 diff);

// "How you feel about <name>: you dislike and distrust them. <their own sentence>"
// Empty when disabled or when the bot has no strong feeling about them.
std::string Regard_WordsFor(Player* bot, Player* other);

// The bot's strongest feelings about other people (excluding `about`), as a
// short prompt section. Empty when there are none.
std::string Regard_PromptSection(Player* bot, Player* about);

// Company words (plan 14 B3): what the bot knows of its company's seats, holdings and rivals, who holds
// the land it stands in, and (when `other` belongs to another company) how the two companies stand.
// `always` skips the OllamaChat.Regard.CompanyChance roll. Empty when disabled or nothing applies.
std::string Regard_CompanySection(Player* bot, Player* other, bool always);

// Rumours (plan 19): one thing the bot's faction is saying in the land it stands in, from chronicle_rumour
// (written by chronicler.py, loaded on the regard thread). `always` skips the
// OllamaChat.Chronicle.RumourChance roll. Empty when disabled or nothing is going around.
std::string Chronicle_RumourSection(Player* bot, bool always);

// Market talk (plan 17 E.4): one thing people are saying about the market in the city the bot stands in, from
// market_word (written by market.py, loaded on the regard thread). `trade` rolls OllamaChat.Market.TradeChance
// (a Trade-channel line), otherwise OllamaChat.Market.Chance. Empty when disabled, outside a market, or no roll.
std::string Market_Section(Player* bot, bool trade);

#endif // MOD_OLLAMA_CHAT_SENTIMENT_H
