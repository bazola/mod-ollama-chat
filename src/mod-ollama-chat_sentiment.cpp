#include "mod-ollama-chat_sentiment.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat_api.h"
#include "mod-ollama-chat_dispatch.h"
#include "mod-ollama-chat-utilities.h"
#include "Log.h"
#include "DatabaseEnv.h"
#include "Player.h"
#include "Random.h"
#include "World.h"
#include <fmt/core.h>
#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

float GetBotPlayerSentiment(uint64_t botGuid, uint64_t playerGuid)
{
    if (!g_EnableSentimentTracking)
        return g_SentimentDefaultValue;

    std::lock_guard<std::mutex> lock(g_SentimentMutex);
    
    auto botIt = g_BotPlayerSentiments.find(botGuid);
    if (botIt != g_BotPlayerSentiments.end())
    {
        auto playerIt = botIt->second.find(playerGuid);
        if (playerIt != botIt->second.end())
        {
            return playerIt->second;
        }
    }
    
    // Return default value if not found
    return g_SentimentDefaultValue;
}

void SetBotPlayerSentiment(uint64_t botGuid, uint64_t playerGuid, float sentimentValue)
{
    if (!g_EnableSentimentTracking)
        return;

    // Clamp sentiment value to valid range [0.0, 1.0]
    sentimentValue = std::max(0.0f, std::min(1.0f, sentimentValue));
    
    std::lock_guard<std::mutex> lock(g_SentimentMutex);
    g_BotPlayerSentiments[botGuid][playerGuid] = sentimentValue;
    g_DirtySentiments.emplace(botGuid, playerGuid);
    
    if (g_DebugEnabled)
    {
        LOG_INFO("module.ollamachat", "[Ollama Chat] Set sentiment between bot {} and player {} to {:.2f}", 
                 botGuid, playerGuid, sentimentValue);
    }
}

std::string BuildSentimentPrompt(const std::string& message)
{
    if (!g_EnableSentimentTracking || message.empty() || g_SentimentAnalysisPrompt.empty())
        return "";

    return SafeFormat(g_SentimentAnalysisPrompt, fmt::arg("message", message));
}

float AnalyzeMessageSentiment(const std::string& prompt)
{
    if (!g_EnableSentimentTracking || prompt.empty())
        return 0.0f;
    
    if (g_DebugEnabled)
    {
        LOG_INFO("module.ollamachat", "[Ollama Chat] Sentiment analysis prompt: {}", prompt);
    }
    
    // Sentiment is a judgement call and is never shown to players, so this is
    // the one request kind that auto think-mode turns reasoning ON for.
    OllamaApiResult api = QueryOllama(prompt, OllamaRequestKind::Sentiment);
    std::string response = api.ok ? api.text : std::string();

    if (response.empty())
    {
        if (g_DebugEnabled)
            LOG_INFO("module.ollamachat", "[Ollama Chat] Empty sentiment analysis response");
        return 0.0f;
    }
    
    // Convert response to uppercase for comparison
    std::string upperResponse = response;
    std::transform(upperResponse.begin(), upperResponse.end(), upperResponse.begin(), ::toupper);
    
    // Parse the sentiment response
    float adjustment = 0.0f;
    if (upperResponse.find("POSITIVE") != std::string::npos)
    {
        adjustment = g_SentimentAdjustmentStrength;
    }
    else if (upperResponse.find("NEGATIVE") != std::string::npos)
    {
        adjustment = -g_SentimentAdjustmentStrength;
    }
    // NEUTRAL or unrecognized = 0.0f (no change)
    
    if (g_DebugEnabled)
    {
        LOG_INFO("module.ollamachat", "[Ollama Chat] Sentiment analysis: '{}' -> adjustment: {:.2f}", 
                 response, adjustment);
    }
    
    return adjustment;
}

void ApplySentimentAnalysis(uint64_t botGuid, uint64_t playerGuid,
                            const std::string& message, const std::string& prompt)
{
    if (!g_EnableSentimentTracking || message.empty() || prompt.empty())
        return;

    const float currentSentiment = GetBotPlayerSentiment(botGuid, playerGuid);
    const float adjustment       = AnalyzeMessageSentiment(prompt);

    if (adjustment == 0.0f)
        return;

    const float newSentiment = currentSentiment + adjustment;
    SetBotPlayerSentiment(botGuid, playerGuid, newSentiment);

    if (g_DebugEnabled)
    {
        LOG_INFO("module.ollamachat",
                 "[Ollama Chat] Sentiment {:.2f} -> {:.2f} ({:+.2f}) for bot {} toward player {}",
                 currentSentiment, newSentiment, adjustment, botGuid, playerGuid);
    }
}

void UpdateBotPlayerSentiment(Player* bot, Player* player, const std::string& message)
{
    if (!g_EnableSentimentTracking || !bot || !player || message.empty())
        return;

    // Hand off rather than block. This used to run the LLM call inline, on
    // whatever thread happened to be delivering the reply.
    OllamaDispatch_SubmitSentiment(bot->GetGUID().GetRawValue(),
                                   player->GetGUID().GetRawValue(),
                                   message);
}

std::string GetSentimentPromptAddition(Player* bot, Player* player)
{
    if (!g_EnableSentimentTracking || !bot || !player || g_SentimentPromptTemplate.empty())
        return "";

    uint64_t botGuid = bot->GetGUID().GetRawValue();
    uint64_t playerGuid = player->GetGUID().GetRawValue();
    
    float sentimentValue = GetBotPlayerSentiment(botGuid, playerGuid);
    
    return SafeFormat(
        g_SentimentPromptTemplate,
        fmt::arg("player_name", player->GetName()),
        fmt::arg("sentiment_value", sentimentValue)
    );
}

void LoadBotPlayerSentimentsFromDB()
{
    if (!g_EnableSentimentTracking)
        return;

    std::lock_guard<std::mutex> lock(g_SentimentMutex);
    g_BotPlayerSentiments.clear();
    g_DirtySentiments.clear();
    
    QueryResult result = CharacterDatabase.Query("SELECT bot_guid, player_guid, sentiment_value FROM mod_ollama_chat_bot_player_sentiments");
    
    if (!result)
    {
        LOG_INFO("module.ollamachat", "[Ollama Chat] No existing sentiment data found in database");
        return;
    }
    
    uint32_t count = 0;
    do
    {
        Field* fields = result->Fetch();
        uint64_t botGuid = fields[0].Get<uint64_t>();
        uint64_t playerGuid = fields[1].Get<uint64_t>();
        float sentimentValue = fields[2].Get<float>();
        
        g_BotPlayerSentiments[botGuid][playerGuid] = sentimentValue;
        count++;
        
    } while (result->NextRow());
    
    LOG_INFO("module.ollamachat", "[Ollama Chat] Loaded {} sentiment records from database", count);
}

void SaveBotPlayerSentimentsToDB()
{
    if (!g_EnableSentimentTracking)
        return;

    // Only the pairs that actually moved since the last save. This used to
    // rewrite every pair the server had ever tracked, every interval, whether
    // or not a single sentiment had changed.
    std::vector<std::pair<std::pair<uint64_t, uint64_t>, float>> changed;

    {
        std::lock_guard<std::mutex> lock(g_SentimentMutex);

        if (g_DirtySentiments.empty())
            return;

        changed.reserve(g_DirtySentiments.size());

        for (const auto& [botGuid, playerGuid] : g_DirtySentiments)
        {
            auto botIt = g_BotPlayerSentiments.find(botGuid);
            if (botIt == g_BotPlayerSentiments.end())
                continue;

            auto playerIt = botIt->second.find(playerGuid);
            if (playerIt == botIt->second.end())
                continue;    // reset out from under us; the reset did its own delete

            changed.push_back({ { botGuid, playerGuid }, playerIt->second });
        }

        g_DirtySentiments.clear();
    }

    if (changed.empty())
        return;

    // INSERT ... ON DUPLICATE KEY UPDATE rather than REPLACE INTO. REPLACE is
    // a delete plus an insert: it rewrites the row, all three secondary
    // indexes and the auto-increment counter even when only the float moved.
    CharacterDatabaseTransaction trans = CharacterDatabase.BeginTransaction();

    for (const auto& [key, sentimentValue] : changed)
    {
        trans->Append(SafeFormat(
            "INSERT INTO mod_ollama_chat_bot_player_sentiments "
            "(bot_guid, player_guid, sentiment_value) VALUES ({}, {}, {:.3f}) "
            "ON DUPLICATE KEY UPDATE sentiment_value = VALUES(sentiment_value)",
            key.first, key.second, sentimentValue));
    }

    CharacterDatabase.CommitTransaction(trans);

    if (g_DebugEnabled)
    {
        LOG_INFO("module.ollamachat", "[Ollama Chat] Saved {} changed sentiment record(s) to database",
                 static_cast<uint32_t>(changed.size()));
    }
}

void InitializeSentimentTracking()
{
    if (!g_EnableSentimentTracking)
    {
        LOG_INFO("module.ollamachat", "[Ollama Chat] Sentiment tracking is disabled");
        return;
    }
    
    LOG_INFO("module.ollamachat", "[Ollama Chat] Initializing sentiment tracking system...");
    
    // Load existing sentiment data from database
    LoadBotPlayerSentimentsFromDB();
    
    // Initialize the last save time
    g_LastSentimentSaveTime = time(nullptr);
    
    LOG_INFO("module.ollamachat", "[Ollama Chat] Sentiment tracking system initialized");
}

// --------------------------------------------------------------------------
// Regard (local patch, plan 14)
// --------------------------------------------------------------------------
// How a bot feels about each person it has dealt with, player or bot. Scored
// outside the worldserver by /opt/wow/regard/regard.py into `regard`; this side
// only reads. The table is loaded on a background thread and swapped in whole,
// so building a prompt costs a mutex and a map lookup.

namespace
{
    struct RegardEntry
    {
        uint32_t    otherGuid;
        float       score;          // -100 hatred .. 100 devotion
        std::string otherName;
        std::string description;    // one sentence in the bot's own words, may be empty
        std::string aside;          // regard_aside (plans/18 P3): e.g. a sponsor's mind to bring them in, may be empty
        // What actually passed between the two of them, newest first, in regard.py's words ("slew Edwin
        // VanCleef together", "they called you an orcish warlock, mocking your race"). The description
        // above says what the bot feels; this says what happened. Only the second one can be brought up.
        std::vector<std::string> passed;
    };

    // bot guid (counter) -> entries, strongest feeling first
    using RegardTable = std::unordered_map<uint32_t, std::vector<RegardEntry>>;

    std::mutex                          g_RegardMutex;
    std::shared_ptr<const RegardTable>  g_RegardTable;
    std::atomic<bool>                   g_RegardLoading{ false };

    // regard_log holds every moment regard.py scored, in words, per pair: it is the only record anywhere
    // of what two people have actually been through together, and nothing has ever read it into a prompt.
    // Loaded whole on the regard thread and attached to the pairs already kept, so a prompt costs no query.
    void LoadPassedBetween(RegardTable& table)
    {
        if (g_RegardPassedPerPrompt == 0)
            return;
        if (!CharacterDatabase.Query("SELECT 1 FROM information_schema.tables "
                                     "WHERE table_schema = DATABASE() AND table_name = 'regard_log'"))
            return;

        // Index the pairs we kept, so a log row for a pair below MinStrength costs nothing.
        std::unordered_map<uint64_t, RegardEntry*> byPair;
        for (auto& [botGuid, entries] : table)
            for (RegardEntry& e : entries)
                byPair[(uint64_t(botGuid) << 32) | e.otherGuid] = &e;

        // Newest first, and only recent moments: regard_log is append-only and never pruned, and this runs
        // every RefreshSeconds. A pair fills up after PassedPerPrompt rows and the rest are skipped, but the
        // scan itself would grow without bound. Old moments are the wrong ones to name anyway — the feeling
        // they caused has already decayed into the score.
        QueryResult result = CharacterDatabase.Query(SafeFormat(
            "SELECT bot_guid, other_guid, reason FROM regard_log "
            "WHERE reason <> '' AND ts > NOW() - INTERVAL {} DAY ORDER BY id DESC", g_RegardPassedDays));
        if (!result)
            return;

        do
        {
            Field* f = result->Fetch();
            auto it = byPair.find((uint64_t(f[0].Get<uint32_t>()) << 32) | f[1].Get<uint32_t>());
            if (it == byPair.end() || it->second->passed.size() >= g_RegardPassedPerPrompt)
                continue;
            std::string reason = f[2].Get<std::string>();
            // regard.py scores the same exchange from both sides and repeats a standing reason as a pair
            // goes on; the same sentence twice in a prompt reads as an obsession, not a memory.
            if (std::find(it->second->passed.begin(), it->second->passed.end(), reason) == it->second->passed.end())
                it->second->passed.push_back(std::move(reason));
        } while (result->NextRow());
    }

    void LoadRegardTable()
    {
        auto table = std::make_shared<RegardTable>();

        if (CharacterDatabase.Query("SELECT 1 FROM information_schema.tables "
                                    "WHERE table_schema = DATABASE() AND table_name = 'regard'"))
        {
            // regard_aside (plans/18 P3) is kept by regard.py; a person with an aside is loaded however faint the
            // feeling, so the clause is never dropped.
            bool const asides = bool(CharacterDatabase.Query("SELECT 1 FROM information_schema.tables "
                                                             "WHERE table_schema = DATABASE() AND table_name = 'regard_aside'"));
            QueryResult result = asides
                ? CharacterDatabase.Query(SafeFormat(
                      "SELECT r.bot_guid, r.other_guid, r.score, c.name, COALESCE(r.description, ''), COALESCE(a.words, '') "
                      "FROM regard r JOIN characters c ON c.guid = r.other_guid "
                      "LEFT JOIN regard_aside a ON a.bot_guid = r.bot_guid AND a.other_guid = r.other_guid "
                      "WHERE ABS(r.score) >= {} OR a.bot_guid IS NOT NULL ORDER BY r.bot_guid, ABS(r.score) DESC",
                      g_RegardMinStrength))
                : CharacterDatabase.Query(SafeFormat(
                      "SELECT r.bot_guid, r.other_guid, r.score, c.name, COALESCE(r.description, ''), '' "
                      "FROM regard r JOIN characters c ON c.guid = r.other_guid "
                      "WHERE ABS(r.score) >= {} ORDER BY r.bot_guid, ABS(r.score) DESC",
                      g_RegardMinStrength));

            if (result)
            {
                do
                {
                    Field* f = result->Fetch();
                    (*table)[f[0].Get<uint32_t>()].push_back({ f[1].Get<uint32_t>(), f[2].Get<float>(),
                                                               f[3].Get<std::string>(), f[4].Get<std::string>(),
                                                               f[5].Get<std::string>(), {} });
                } while (result->NextRow());
            }

            LoadPassedBetween(*table);
        }

        std::lock_guard<std::mutex> lock(g_RegardMutex);
        g_RegardTable = std::move(table);
    }

    std::shared_ptr<const RegardTable> RegardSnapshot()
    {
        std::lock_guard<std::mutex> lock(g_RegardMutex);
        return g_RegardTable;
    }

    // Words, never numbers: the standing rule is no figures in bot context.
    // Keep in step with regard.py's words().
    char const* RegardWords(float score)
    {
        if (score <= -60.0f) return "you despise them";
        if (score <= -30.0f) return "you dislike and distrust them";
        if (score <= -10.0f) return "you are wary of them";
        if (score < 10.0f)   return "you feel little either way about them";
        if (score < 30.0f)   return "you are on good terms with them";
        if (score < 60.0f)   return "you like and trust them";
        return "you would stand by them through anything";
    }

    std::string RegardLine(RegardEntry const& e)
    {
        std::string line = e.otherName + ": " + RegardWords(e.score) + ".";
        if (!e.description.empty())
            line += " " + e.description;
        if (!e.aside.empty())
            line += " " + e.aside;
        return line;
    }

    // Company words (plan 14 B3). regard.py rewrites guild_words and land_words every cycle; they load
    // on the same thread and timer as the regard table.
    struct CompanyWords
    {
        std::unordered_map<uint32_t, std::string> guilds;   // guild id -> what members know of their company
        std::unordered_map<uint32_t, std::string> lands;    // zone id -> who holds the land
        std::unordered_map<uint32_t, std::string> names;    // guild id -> name
        std::unordered_map<uint64_t, float>       stances;  // PairKey(guild, guild) -> -100 .. 100
    };

    std::shared_ptr<const CompanyWords> g_CompanyWords;   // guarded by g_RegardMutex

    uint64_t PairKey(uint32_t a, uint32_t b)
    {
        return (uint64_t(std::min(a, b)) << 32) | std::max(a, b);
    }

    bool TableExists(char const* name)
    {
        return bool(CharacterDatabase.Query(SafeFormat(
            "SELECT 1 FROM information_schema.tables WHERE table_schema = DATABASE() AND table_name = '{}'", name)));
    }

    void LoadCompanyWords()
    {
        auto words = std::make_shared<CompanyWords>();

        auto load = [](char const* table, char const* sql, auto&& add)
        {
            if (!TableExists(table))
                return;
            if (QueryResult result = CharacterDatabase.Query(sql))
            {
                do
                {
                    add(result->Fetch());
                } while (result->NextRow());
            }
        };

        load("guild_words", "SELECT guildid, words FROM guild_words",
             [&](Field* f) { words->guilds[f[0].Get<uint32_t>()] = f[1].Get<std::string>(); });
        load("land_words", "SELECT zone_id, words FROM land_words",
             [&](Field* f) { words->lands[f[0].Get<uint32_t>()] = f[1].Get<std::string>(); });
        load("guild", "SELECT guildid, name FROM guild",
             [&](Field* f) { words->names[f[0].Get<uint32_t>()] = f[1].Get<std::string>(); });
        load("guild_relation", "SELECT guild_a, guild_b, stance FROM guild_relation",
             [&](Field* f) { words->stances[PairKey(f[0].Get<uint32_t>(), f[1].Get<uint32_t>())] = f[2].Get<float>(); });

        std::lock_guard<std::mutex> lock(g_RegardMutex);
        g_CompanyWords = std::move(words);
    }

    // Keep in step with regard.py's stance_words() and COMPANY_TALK. Nothing for indifference.
    char const* StanceTalk(float stance)
    {
        if (stance <= -60.0f) return "their company and yours are in a blood feud";
        if (stance <= -30.0f) return "their company and yours are bitter rivals";
        if (stance <= -10.0f) return "their company and yours are wary of each other";
        if (stance < 10.0f)   return nullptr;
        if (stance < 30.0f)   return "their company and yours are on good terms";
        if (stance < 60.0f)   return "their company and yours are friends";
        return "their company and yours are sworn allies";
    }
}

// Rumours (plan 19). chronicler.py writes chronicle_rumour after every watch: what each faction's people are
// saying in each land, plain where it happened and garbled further off. Loaded on the regard thread and timer.
namespace
{
    constexpr std::size_t RUMOURS_PER_PLACE = 6;   // nearest news first, then the newest

    uint64_t RumourKey(uint32_t zoneId, uint32_t team)
    {
        return (uint64_t(zoneId) << 8) | team;
    }

    using RumourMap = std::unordered_map<uint64_t, std::vector<std::string>>;   // RumourKey -> words

    std::shared_ptr<const RumourMap> g_Rumours;   // guarded by g_RegardMutex

    void LoadRumours()
    {
        if (!TableExists("chronicle_rumour"))
            return;

        auto rumours = std::make_shared<RumourMap>();
        if (QueryResult result = CharacterDatabase.Query(
                "SELECT zone_id, team, words FROM chronicle_rumour WHERE expires_at > NOW() ORDER BY distance, id DESC"))
        {
            do
            {
                Field* f = result->Fetch();
                auto& place = (*rumours)[RumourKey(f[0].Get<uint32_t>(), f[1].Get<uint8_t>())];
                if (place.size() < RUMOURS_PER_PLACE)
                    place.push_back(f[2].Get<std::string>());
            } while (result->NextRow());
        }

        std::lock_guard<std::mutex> lock(g_RegardMutex);
        g_Rumours = std::move(rumours);
    }

    // Market talk (plan 17 E.4). market.py writes market_word from the auction houses: what is plentiful, scarce
    // or selling in each city's market, in words. Team 2 rows are the neutral houses, heard by both teams.
    std::shared_ptr<const RumourMap> g_MarketWords;   // RumourKey -> words, guarded by g_RegardMutex

    void LoadMarketWords()
    {
        if (!TableExists("market_word"))
            return;

        auto market = std::make_shared<RumourMap>();
        if (QueryResult result = CharacterDatabase.Query(
                "SELECT zone_id, team, words FROM market_word WHERE expires_at > NOW() ORDER BY id DESC"))
        {
            do
            {
                Field* f = result->Fetch();
                const uint32_t zoneId = f[0].Get<uint32_t>();
                const uint8_t team = f[1].Get<uint8_t>();
                for (uint32_t t = 0; t < 2; ++t)
                {
                    if (team != t && team != 2)
                        continue;
                    auto& place = (*market)[RumourKey(zoneId, t)];
                    if (place.size() < RUMOURS_PER_PLACE)
                        place.push_back(f[2].Get<std::string>());
                }
            } while (result->NextRow());
        }

        std::lock_guard<std::mutex> lock(g_RegardMutex);
        g_MarketWords = std::move(market);
    }
}

void Regard_Tick(uint32 diff)
{
    static uint32 timer = 0;    // 0: load on the first tick after enabling

    if (!g_RegardEnable)
        return;

    if (timer > diff)
    {
        timer -= diff;
        return;
    }

    timer = std::max<uint32>(10, g_RegardRefreshSeconds) * 1000;

    if (World::IsStopped() || g_RegardLoading.exchange(true))
        return;

    std::thread([]
    {
        LoadRegardTable();
        if (g_RegardCompanyWords)
            LoadCompanyWords();
        if (g_ChronicleRumours)
            LoadRumours();
        if (g_MarketTalk)
            LoadMarketWords();
        g_RegardLoading = false;
    }).detach();
}

std::string Regard_WordsFor(Player* bot, Player* other)
{
    if (!g_RegardEnable || !bot || !other)
        return "";

    auto table = RegardSnapshot();
    if (!table)
        return "";

    auto it = table->find(bot->GetGUID().GetCounter());
    if (it == table->end())
        return "";

    const uint32_t otherGuid = other->GetGUID().GetCounter();
    for (RegardEntry const& e : it->second)
        if (e.otherGuid == otherGuid)
        {
            std::string out = "How you feel about " + RegardLine(e);
            if (!e.passed.empty())
            {
                out += " What has passed between you, most recent first:";
                for (std::string const& p : e.passed)
                    out += " " + p + ";";
                out.back() = '.';
            }
            return out;
        }

    return "";
}

std::string Regard_PromptSection(Player* bot, Player* about)
{
    if (!g_RegardEnable || !bot || g_RegardMaxPerPrompt == 0)
        return "";

    auto table = RegardSnapshot();
    if (!table)
        return "";

    auto it = table->find(bot->GetGUID().GetCounter());
    if (it == table->end())
        return "";

    const uint32_t aboutGuid = about ? about->GetGUID().GetCounter() : 0;

    std::string lines;
    uint32_t taken = 0;
    for (RegardEntry const& e : it->second)
    {
        if (e.otherGuid == aboutGuid)
            continue;
        if (taken >= g_RegardMaxPerPrompt)
            break;
        lines += " - " + RegardLine(e) + "\n";
        ++taken;
    }

    if (lines.empty())
        return "";

    return "\nPeople you feel strongly about (bring them up only if it fits):\n" + lines;
}

std::string Regard_CompanySection(Player* bot, Player* other, bool always)
{
    if (!g_RegardEnable || !g_RegardCompanyWords || !bot)
        return "";

    if (!always && urand(0, 99) >= g_RegardCompanyChance)
        return "";

    std::shared_ptr<const CompanyWords> words;
    {
        std::lock_guard<std::mutex> lock(g_RegardMutex);
        words = g_CompanyWords;
    }
    if (!words)
        return "";

    std::string text;
    const uint32_t guildId = bot->GetGuildId();

    if (guildId)
    {
        auto it = words->guilds.find(guildId);
        if (it != words->guilds.end())
            text += it->second + "\n";
    }

    auto land = words->lands.find(bot->GetZoneId());
    if (land != words->lands.end())
        text += land->second + "\n";

    if (other && other != bot && guildId && other->GetGuildId() && other->GetGuildId() != guildId)
    {
        auto stance = words->stances.find(PairKey(guildId, other->GetGuildId()));
        auto name = words->names.find(other->GetGuildId());
        if (stance != words->stances.end() && name != words->names.end())
            if (char const* talk = StanceTalk(stance->second))
                text += other->GetName() + " is of " + name->second + ": " + talk + ".\n";
    }

    if (text.empty())
        return "";

    return "\nYour company and the lands around you (bring it up only if it fits):\n" + text;
}

std::string Chronicle_RumourSection(Player* bot, bool always)
{
    if (!g_RegardEnable || !g_ChronicleRumours || !bot)
        return "";

    if (!always && urand(0, 99) >= g_ChronicleRumourChance)
        return "";

    std::shared_ptr<const RumourMap> rumours;
    {
        std::lock_guard<std::mutex> lock(g_RegardMutex);
        rumours = g_Rumours;
    }
    if (!rumours)
        return "";

    auto it = rumours->find(RumourKey(bot->GetZoneId(), uint32_t(bot->GetTeamId())));
    if (it == rumours->end() || it->second.empty())
        return "";

    return "\nWord going around here (pass it on in your own words, only if it fits):\n"
        + it->second[urand(0, uint32(it->second.size() - 1))] + "\n";
}

std::string Market_Section(Player* bot, bool trade)
{
    if (!g_RegardEnable || !g_MarketTalk || !bot)
        return "";

    if (urand(0, 99) >= (trade ? g_MarketTradeChance : g_MarketChance))
        return "";

    std::shared_ptr<const RumourMap> market;
    {
        std::lock_guard<std::mutex> lock(g_RegardMutex);
        market = g_MarketWords;
    }
    if (!market)
        return "";

    auto it = market->find(RumourKey(bot->GetZoneId(), uint32_t(bot->GetTeamId())));
    if (it == market->end() || it->second.empty())
        return "";

    return "\nTalk at the market here: " + it->second[urand(0, uint32(it->second.size() - 1))] + "\n"
        "(Mention it naturally and in character, as something you saw or heard at the stalls; "
        "never give prices, counts or other numbers.)\n";
}
