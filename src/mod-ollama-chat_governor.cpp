#include "mod-ollama-chat_governor.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat-utilities.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <deque>
#include <iterator>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace
{
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    std::mutex g_mutex;

    using GramCounts = std::unordered_map<std::string, int>;

    struct Utterance
    {
        std::string           normalized;
        std::string           opener;
        std::set<std::string> tokens;    // content words, stopwords dropped
        GramCounts            grams;     // character trigrams
        double                gramNorm = 0.0;   // sqrt(sum of squares)
        TimePoint             when;

        // Each sentence of the line, normalized. Whole-line scoring cannot see
        // a catchphrase riding along at the end of an otherwise new answer.
        std::vector<std::string> sentences;
    };

    // A line as it was said, with who said it. Kept apart from Utterance
    // because that one is built for comparison -- normalized, speakerless --
    // and this one is built to be read back in a prompt.
    struct SpokenLine
    {
        std::string speaker;
        std::string text;
        TimePoint   when;
    };

    // Enough to answer "who raised this subject" without keeping a transcript.
    constexpr size_t kMaxScopeLines = 16;

    struct BotState
    {
        TimePoint             lastSend{};
        TimePoint             lastEvent{};
        std::deque<Utterance> history;
        std::unordered_map<uint64_t, TimePoint> emoteReactions;  // player guid -> last

        // Who this bot is mid-conversation with, and where: player guid ->
        // scope key -> when it last answered them there. While that is fresh
        // the next thing that person says in that scope is a continuation, not
        // ambient chatter, so it bypasses the pacing cooldowns.
        std::unordered_map<uint64_t, std::unordered_map<std::string, TimePoint>> conversations;
    };

    // Who a person is mid-exchange with in one scope, and how deep it has run.
    // Kept per scope rather than per bot because the question routing asks is
    // "who holds the thread here", and the asker does not yet know the bot.
    struct ThreadHolder
    {
        uint64_t  botGuid = 0;
        TimePoint lastLineAt{};
        uint32_t  turns    = 0;
    };

    struct ScopeState
    {
        TimePoint             lastHuman{};
        TimePoint             lastSend{};
        std::deque<Utterance> history;
        std::deque<TimePoint> sendTimes;

        // What was actually said here, and by whom.
        std::deque<SpokenLine> lines;

        // Player guid -> the bot they are mid-exchange with here.
        std::unordered_map<uint64_t, ThreadHolder> holders;

        // The staleness end condition (plan 25 item 62's build half). A chain
        // that has stopped saying anything new should end, and until now
        // nothing could end one: repetition suppressed the offending LINE and
        // the next bot simply tried again, so the exchange decayed into echo
        // instead of concluding -- measured 41 runs of three or more lines, the
        // longest nine over twelve minutes.
        //
        // `staleHits` counts consecutive lines here that said nothing new;
        // anything new resets it. At the threshold `staleUntil` is set and
        // bot-to-bot replies in this scope stop until it passes. Ambient and
        // event lines are deliberately NOT gated: they seed at chainDepth 0 and
        // a genuinely new subject is exactly what should be allowed to follow a
        // dead one. The player is never consulted -- the end condition is
        // staleness only, decided 2026-09-18 (§38 Q1), because ending on a
        // person's departure would restore the audience brake by another name.
        uint32_t  staleHits = 0;
        TimePoint staleUntil{};
    };

    std::unordered_map<uint64_t, BotState>    g_bots;
    std::unordered_map<std::string, ScopeState> g_scopes;
    std::deque<TimePoint>                     g_globalSends;

    GovernorStats g_stats{};

    inline double SecondsSince(TimePoint t, TimePoint now)
    {
        if (t.time_since_epoch().count() == 0)
            return 1e9;   // never happened
        return std::chrono::duration_cast<std::chrono::milliseconds>(now - t).count() / 1000.0;
    }

    // How long this holder survives silence. An exchange that has run several
    // turns earns a longer pause than an opening line does -- people stop to
    // fight something and come back to what they were saying -- and the cap
    // stops a long conversation owning the thread indefinitely.
    inline double HolderWindowFor(const ThreadHolder& h)
    {
        const double bonus =
            std::min<double>(double(h.turns) * double(g_HolderTurnBonusSeconds),
                             double(g_HolderMaxBonusSeconds));
        return double(g_HolderWindowSeconds) + bonus;
    }

    void TrimWindow(std::deque<TimePoint>& times, TimePoint now, double windowSec)
    {
        while (!times.empty() && SecondsSince(times.front(), now) > windowSec)
            times.pop_front();
    }

    // ---- text normalisation ---------------------------------------------

    const std::unordered_set<std::string>& Stopwords()
    {
        static const std::unordered_set<std::string> s = {
            "a","an","the","and","or","but","if","of","to","in","on","at","for",
            "is","are","was","were","be","been","am","i","you","he","she","it",
            "we","they","me","my","your","this","that","these","those","so",
            "just","really","very","gonna","got","get","do","does","did","have",
            "has","had","will","would","can","could","should","not","no","yes"
        };
        return s;
    }

    std::string NormalizeText(const std::string& text)
    {
        std::string out;
        out.reserve(text.size());
        for (char ch : text)
        {
            unsigned char c = static_cast<unsigned char>(ch);
            if (std::isalnum(c))
                out.push_back(static_cast<char>(std::tolower(c)));
            else if (!out.empty() && out.back() != ' ')
                out.push_back(' ');
        }
        while (!out.empty() && out.back() == ' ')
            out.pop_back();
        return out;
    }

    std::vector<std::string> Tokenize(const std::string& normalized, bool dropStopwords)
    {
        std::vector<std::string> tokens;
        size_t start = 0;
        while (start < normalized.size())
        {
            size_t end = normalized.find(' ', start);
            if (end == std::string::npos)
                end = normalized.size();
            if (end > start)
            {
                std::string tok = normalized.substr(start, end - start);
                if (!dropStopwords || !Stopwords().count(tok))
                    tokens.push_back(std::move(tok));
            }
            start = end + 1;
        }
        return tokens;
    }

    std::string OpenerOf(const std::string& normalized)
    {
        auto tokens = Tokenize(normalized, false);
        std::string opener;
        for (size_t i = 0; i < tokens.size() && i < 3; ++i)
        {
            if (!opener.empty())
                opener.push_back(' ');
            opener += tokens[i];
        }
        return opener;
    }

    std::set<std::string> BuildTokenSet(const std::string& normalized)
    {
        std::set<std::string> out;
        for (auto& t : Tokenize(normalized, true))
            out.insert(std::move(t));
        return out;
    }

    GramCounts BuildGrams(const std::string& s)
    {
        GramCounts m;
        if (s.size() < 3)
        {
            if (!s.empty())
                m[s] = 1;
            return m;
        }
        for (size_t i = 0; i + 3 <= s.size(); ++i)
            ++m[s.substr(i, 3)];
        return m;
    }

    double GramNorm(const GramCounts& g)
    {
        double n = 0.0;
        for (const auto& [gram, c] : g)
            n += double(c) * c;
        return std::sqrt(n);
    }

    float JaccardOf(const std::set<std::string>& sa, const std::set<std::string>& sb)
    {
        if (sa.empty() || sb.empty())
            return 0.0f;

        size_t inter = 0;
        for (const auto& t : sa)
            if (sb.count(t))
                ++inter;

        const size_t uni = sa.size() + sb.size() - inter;
        return uni == 0 ? 0.0f : static_cast<float>(inter) / static_cast<float>(uni);
    }

    float CosineOf(const GramCounts& ga, double na, const GramCounts& gb, double nb)
    {
        if (na <= 0.0 || nb <= 0.0)
            return 0.0f;

        // Iterate the smaller map.
        //
        // Not named `small`: the Windows SDK's rpcndr.h has `#define small
        // char`, so on any translation unit that ends up including it this
        // read as `const GramCounts& char = ...` and failed to compile. It
        // built here only because our include chain happens not to pull that
        // header in -- which is luck, not design.
        const GramCounts& smaller = ga.size() <= gb.size() ? ga : gb;
        const GramCounts& larger  = ga.size() <= gb.size() ? gb : ga;

        double dot = 0.0;
        for (const auto& [gram, count] : smaller)
        {
            auto it = larger.find(gram);
            if (it != larger.end())
                dot += double(count) * it->second;
        }

        return static_cast<float>(dot / (na * nb));
    }

    void TrimHistory(std::deque<Utterance>& hist, size_t maxSize)
    {
        while (hist.size() > maxSize)
            hist.pop_front();
    }

    // The sentence spans of a line, raw text preserved, so a sentence that is
    // kept goes back exactly as the model wrote it. Terminators run together
    // ("...!?") and a closing quote or bracket belongs to the sentence it ends.
    std::vector<std::string> RawSentences(const std::string& text)
    {
        std::vector<std::string> out;
        size_t start = 0;

        for (size_t i = 0; i < text.size(); ++i)
        {
            const char ch = text[i];
            if (ch != '.' && ch != '!' && ch != '?')
                continue;

            size_t end = i + 1;
            while (end < text.size() &&
                   (text[end] == '.' || text[end] == '!' || text[end] == '?' ||
                    text[end] == '"' || text[end] == '\'' || text[end] == ')'))
                ++end;

            out.push_back(text.substr(start, end - start));

            while (end < text.size() && std::isspace(static_cast<unsigned char>(text[end])))
                ++end;

            start = end;
            i     = end > 0 ? end - 1 : end;
        }

        // A trailing fragment with no terminator is still a sentence: the model
        // is told to keep replies short and often ends without one.
        if (start < text.size() && !NormalizeText(text.substr(start)).empty())
            out.push_back(text.substr(start));

        return out;
    }

    // Normalized sentences long enough to be a tic rather than an "Aye." Short
    // interjections repeat honestly, and suppressing them makes a bot evasive.
    std::vector<std::string> SentenceKeys(const std::string& text)
    {
        std::vector<std::string> keys;

        for (const std::string& raw : RawSentences(text))
        {
            const std::string norm = NormalizeText(raw);
            if (norm.empty())
                continue;
            if (Tokenize(norm, false).size() < size_t(g_SentenceRepeatMinWords))
                continue;
            keys.push_back(norm);
        }

        return keys;
    }
}

// --------------------------------------------------------------------------

std::string Governor_MakeScopeKey(const char* sourceName, uint32_t channelId,
                                  const std::string& channelName,
                                  uint32_t guildId, uint32_t groupOrZoneId)
{
    std::string key = sourceName ? sourceName : "unknown";

    if (!channelName.empty())
    {
        key += "#";
        key += channelName;
        key += ":";
        key += std::to_string(channelId);
    }
    else if (guildId)
    {
        key += "#g";
        key += std::to_string(guildId);
    }
    else
    {
        key += "#z";
        key += std::to_string(groupOrZoneId);
    }

    return key;
}

// --- the audience rule ----------------------------------------------------

void Governor_NoteHumanMessage(const std::string& scopeKey)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    g_scopes[scopeKey].lastHuman = Clock::now();
}

bool Governor_HasRecentHuman(const std::string& scopeKey)
{
    if (!g_RequireRecentHuman)
        return true;

    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_scopes.find(scopeKey);
    if (it == g_scopes.end())
    {
        ++g_stats.blockedNoAudience;
        return false;
    }

    if (SecondsSince(it->second.lastHuman, Clock::now()) <= double(g_HumanWindowSeconds))
        return true;

    ++g_stats.blockedNoAudience;
    return false;
}

// --- chain depth ----------------------------------------------------------

bool Governor_ChainDepthAllowed(uint8_t depth)
{
    if (depth < g_MaxChainDepth)
        return true;

    std::lock_guard<std::mutex> lock(g_mutex);
    ++g_stats.blockedChainDepth;
    return false;
}

// --- the staleness end condition ------------------------------------------
//
// The mark itself is taken in Governor_RecordUtterance, where the candidate's
// tokens and grams are already built and where every recorded line passes. This
// is only the reader.

bool Governor_ScopeIsStale(const std::string& scopeKey)
{
    if (g_StaleChainHits == 0)
        return false;

    std::lock_guard<std::mutex> lock(g_mutex);

    auto it = g_scopes.find(scopeKey);
    if (it == g_scopes.end())
        return false;

    if (it->second.staleUntil.time_since_epoch().count() == 0)
        return false;
    if (Clock::now() >= it->second.staleUntil)
        return false;

    ++g_stats.blockedStale;
    return true;
}

uint32_t Governor_ApplyChainDecay(uint32_t baseChancePct, uint8_t depth)
{
    if (depth == 0 || g_ChainChanceDecayPct >= 100)
        return baseChancePct;

    double chance = baseChancePct;
    for (uint8_t i = 0; i < depth; ++i)
        chance = chance * double(g_ChainChanceDecayPct) / 100.0;

    return static_cast<uint32_t>(chance + 0.5);
}

// --- open conversations ---------------------------------------------------

// Record that this bot just answered this player here. Call it only when the
// addressee is a real person: an open conversation bypasses pacing, and
// letting bots open one with each other is how a bot-to-bot loop escapes
// every brake in this file.
void Governor_NoteConversation(ObjectGuid botGuid, ObjectGuid playerGuid,
                               const std::string& scopeKey)
{
    if (!playerGuid || g_ConversationWindowSeconds == 0)
        return;

    std::lock_guard<std::mutex> lock(g_mutex);
    g_bots[botGuid.GetRawValue()].conversations[playerGuid.GetRawValue()][scopeKey] = Clock::now();
}

bool Governor_InConversation(ObjectGuid botGuid, ObjectGuid playerGuid,
                             const std::string& scopeKey)
{
    if (!playerGuid || g_ConversationWindowSeconds == 0)
        return false;

    std::lock_guard<std::mutex> lock(g_mutex);

    auto botIt = g_bots.find(botGuid.GetRawValue());
    if (botIt == g_bots.end())
        return false;

    auto playerIt = botIt->second.conversations.find(playerGuid.GetRawValue());
    if (playerIt == botIt->second.conversations.end())
        return false;

    auto scopeIt = playerIt->second.find(scopeKey);
    if (scopeIt == playerIt->second.end())
        return false;

    return SecondsSince(scopeIt->second, Clock::now()) <= double(g_ConversationWindowSeconds);
}

// --- the thread holder ----------------------------------------------------

void Governor_NoteThreadHolder(ObjectGuid botGuid, ObjectGuid playerGuid,
                               const std::string& scopeKey)
{
    if (!botGuid || !playerGuid || scopeKey.empty() || g_HolderWindowSeconds == 0)
        return;

    const TimePoint now = Clock::now();

    std::lock_guard<std::mutex> lock(g_mutex);

    ThreadHolder& h = g_scopes[scopeKey].holders[playerGuid.GetRawValue()];

    // The same bot answering again is the exchange running on, so the thread
    // deepens. A different bot taking over starts a new one at turn one rather
    // than inheriting the old one's earned patience.
    if (h.botGuid == botGuid.GetRawValue() &&
        SecondsSince(h.lastLineAt, now) <= HolderWindowFor(h))
    {
        ++h.turns;
    }
    else
    {
        h.botGuid = botGuid.GetRawValue();
        h.turns   = 1;
    }

    h.lastLineAt = now;
}

uint64_t Governor_ThreadHolder(ObjectGuid playerGuid, const std::string& scopeKey)
{
    if (!playerGuid || scopeKey.empty() || g_HolderWindowSeconds == 0)
        return 0;

    std::lock_guard<std::mutex> lock(g_mutex);

    auto scopeIt = g_scopes.find(scopeKey);
    if (scopeIt == g_scopes.end())
        return 0;

    auto it = scopeIt->second.holders.find(playerGuid.GetRawValue());
    if (it == scopeIt->second.holders.end())
        return 0;

    // Checked on read as well as pruned on the tick: a holder that went stale
    // between ticks must not be handed out as live.
    if (SecondsSince(it->second.lastLineAt, Clock::now()) > HolderWindowFor(it->second))
        return 0;

    return it->second.botGuid;
}

// --- cooldowns and rate limits -------------------------------------------

namespace
{
    bool CheckSendLocked(ObjectGuid botGuid, const std::string& scopeKey,
                         TimePoint now, bool reserve, bool directAddress)
    {
        const uint64_t raw = botGuid.GetRawValue();
        BotState&   bot   = g_bots[raw];
        ScopeState& scope = g_scopes[scopeKey];

        // Pacing, skipped for direct address. A bot that just said something
        // in /say would otherwise sit on its cooldown and silently ignore a
        // whisper, which reads as broken rather than as rate limiting.
        if (!directAddress)
        {
            if (g_BotCooldownSeconds > 0 &&
                SecondsSince(bot.lastSend, now) < double(g_BotCooldownSeconds))
            {
                ++g_stats.blockedCooldown;
                return false;
            }

            if (g_ScopeCooldownSeconds > 0 &&
                SecondsSince(scope.lastSend, now) < double(g_ScopeCooldownSeconds))
            {
                ++g_stats.blockedCooldown;
                return false;
            }

            TrimWindow(scope.sendTimes, now, 60.0);
            if (g_ScopeMessagesPerMinute > 0 &&
                scope.sendTimes.size() >= g_ScopeMessagesPerMinute)
            {
                ++g_stats.blockedRate;
                return false;
            }
        }
        else
        {
            TrimWindow(scope.sendTimes, now, 60.0);
        }

        TrimWindow(g_globalSends, now, 60.0);
        if (g_GlobalMessagesPerMinute > 0 &&
            g_globalSends.size() >= g_GlobalMessagesPerMinute)
        {
            ++g_stats.blockedRate;
            return false;
        }

        if (reserve)
        {
            bot.lastSend   = now;
            scope.lastSend = now;
            scope.sendTimes.push_back(now);
            g_globalSends.push_back(now);
        }

        return true;
    }
}

bool Governor_TryConsumeSend(ObjectGuid botGuid, const std::string& scopeKey,
                             bool directAddress)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    return CheckSendLocked(botGuid, scopeKey, Clock::now(), true, directAddress);
}

bool Governor_CanSend(ObjectGuid botGuid, const std::string& scopeKey,
                      bool directAddress)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    return CheckSendLocked(botGuid, scopeKey, Clock::now(), false, directAddress);
}

// --- repetition -----------------------------------------------------------

float Governor_Similarity(const std::string& a, const std::string& b)
{
    const std::string na = NormalizeText(a);
    const std::string nb = NormalizeText(b);
    if (na.empty() || nb.empty())
        return 0.0f;
    if (na == nb)
        return 1.0f;

    const std::set<std::string> ta = BuildTokenSet(na);
    const std::set<std::string> tb = BuildTokenSet(nb);
    const GramCounts ga = BuildGrams(na);
    const GramCounts gb = BuildGrams(nb);

    return std::max(JaccardOf(ta, tb), CosineOf(ga, GramNorm(ga), gb, GramNorm(gb)));
}

bool Governor_IsRepetitive(ObjectGuid botGuid, const std::string& scopeKey,
                           const std::string& text)
{
    const std::string norm = NormalizeText(text);
    if (norm.empty())
        return false;

    const std::string opener = OpenerOf(norm);

    // Built once for the candidate; every stored line already carries its own.
    const std::set<std::string> candTokens = BuildTokenSet(norm);
    const GramCounts            candGrams  = BuildGrams(norm);
    const double                candNorm   = GramNorm(candGrams);

    std::lock_guard<std::mutex> lock(g_mutex);
    const TimePoint now = Clock::now();

    auto tooSimilar = [&](const std::deque<Utterance>& hist) -> bool
    {
        for (const auto& u : hist)
        {
            if (SecondsSince(u.when, now) > double(g_RepetitionWindowSeconds))
                continue;
            if (u.normalized == norm)
                return true;

            // Cheap reject first: lines of wildly different length cannot be
            // similar enough to matter.
            const size_t la = norm.size(), lb = u.normalized.size();
            const size_t lo = la < lb ? la : lb, hi = la < lb ? lb : la;
            if (hi > 0 && double(lo) / double(hi) < 0.4)
                continue;

            if (JaccardOf(candTokens, u.tokens) >= g_RepetitionSimilarityThreshold)
                return true;
            if (CosineOf(candGrams, candNorm, u.grams, u.gramNorm) >= g_RepetitionSimilarityThreshold)
                return true;
        }
        return false;
    };

    auto botIt = g_bots.find(botGuid.GetRawValue());
    if (botIt != g_bots.end() && tooSimilar(botIt->second.history))
    {
        ++g_stats.blockedRepetition;
        return true;
    }

    auto scopeIt = g_scopes.find(scopeKey);
    if (scopeIt != g_scopes.end())
    {
        if (tooSimilar(scopeIt->second.history))
        {
            ++g_stats.blockedRepetition;
            return true;
        }

        // Opener collision: every bot starting "aye lad" is a distinct problem
        // from every bot saying the same whole line, and whole-line similarity
        // does not catch it.
        if (g_OpenerHistorySize > 0 && !opener.empty())
        {
            uint32_t checked = 0;
            for (auto it = scopeIt->second.history.rbegin();
                 it != scopeIt->second.history.rend() && checked < g_OpenerHistorySize;
                 ++it, ++checked)
            {
                if (SecondsSince(it->when, now) > double(g_RepetitionWindowSeconds))
                    continue;
                if (!it->opener.empty() && it->opener == opener)
                {
                    ++g_stats.blockedRepetition;
                    return true;
                }
            }
        }
    }

    return false;
}

bool Governor_HasOpenerCollision(const std::string& scopeKey, const std::string& text)
{
    if (g_OpenerHistorySize == 0)
        return false;

    const std::string norm = NormalizeText(text);
    if (norm.empty())
        return false;

    const std::string opener = OpenerOf(norm);
    if (opener.empty())
        return false;

    std::lock_guard<std::mutex> lock(g_mutex);
    const TimePoint now = Clock::now();

    auto scopeIt = g_scopes.find(scopeKey);
    if (scopeIt == g_scopes.end())
        return false;

    uint32_t checked = 0;
    for (auto it = scopeIt->second.history.rbegin();
         it != scopeIt->second.history.rend() && checked < g_OpenerHistorySize;
         ++it, ++checked)
    {
        if (SecondsSince(it->when, now) > double(g_RepetitionWindowSeconds))
            continue;
        if (!it->opener.empty() && it->opener == opener)
        {
            ++g_stats.blockedRepetition;
            return true;
        }
    }

    return false;
}

void Governor_RecordUtterance(ObjectGuid botGuid, const std::string& scopeKey,
                              const std::string& text)
{
    const std::string norm = NormalizeText(text);
    if (norm.empty())
        return;

    Utterance u;
    u.normalized = norm;
    u.opener     = OpenerOf(norm);
    u.tokens     = BuildTokenSet(norm);
    u.grams      = BuildGrams(norm);
    u.gramNorm   = GramNorm(u.grams);
    u.when       = Clock::now();
    u.sentences  = SentenceKeys(text);

    std::lock_guard<std::mutex> lock(g_mutex);

    BotState& bot = g_bots[botGuid.GetRawValue()];
    bot.history.push_back(u);
    TrimHistory(bot.history, g_BotHistorySize);

    ScopeState& scope = g_scopes[scopeKey];

    // Did this line say anything new here? (Plan 25 item 62, the end
    // condition.) Scored against the scope's history as it stands, BEFORE this
    // utterance joins it -- compare after and every line matches itself, so
    // every conversation would read as stale on its first line.
    //
    // Scope history only, never the bot's own. One bot repeating itself is the
    // tic that Governor_StripRepeatedSentences and the opener check already
    // answer; it must not end the conversation everyone else is having.
    //
    // Judged here rather than at the repetition check because that one is
    // skipped for direct address, and in a party of eight or fewer every line
    // is direct address -- the party the operator actually plays in is exactly
    // where the echo was measured, so a signal taken from there would have been
    // blind to it. This function is called for every line that is recorded.
    if (g_StaleChainHits > 0)
    {
        bool saidSomethingNew = true;

        for (const auto& prev : scope.history)
        {
            if (SecondsSince(prev.when, u.when) > double(g_RepetitionWindowSeconds))
                continue;

            // Same cheap length reject the repetition scorer uses: lines of
            // wildly different length cannot be near-duplicates.
            const size_t la = u.normalized.size(), lb = prev.normalized.size();
            const size_t lo = la < lb ? la : lb, hi = la < lb ? lb : la;
            if (hi > 0 && double(lo) / double(hi) < 0.4)
                continue;

            if (prev.normalized == u.normalized ||
                JaccardOf(u.tokens, prev.tokens) >= g_RepetitionSimilarityThreshold ||
                CosineOf(u.grams, u.gramNorm, prev.grams, prev.gramNorm) >=
                    g_RepetitionSimilarityThreshold)
            {
                saidSomethingNew = false;
                break;
            }
        }

        if (saidSomethingNew)
        {
            scope.staleHits = 0;              // the chain has somewhere to go
        }
        else if (++scope.staleHits >= g_StaleChainHits)
        {
            // Ended. The window is what stops the next ambient line re-opening
            // the exchange that just died: a new subject may start one, but not
            // instantly, and not off the back of the line that said nothing.
            scope.staleHits  = 0;
            scope.staleUntil = u.when + std::chrono::seconds(g_StaleQuietSeconds);
        }
    }

    scope.history.push_back(std::move(u));
    TrimHistory(scope.history, g_ScopeHistorySize);
}

std::string Governor_StripRepeatedSentences(ObjectGuid botGuid, const std::string& text)
{
    if (g_SentenceRepeatMinWords == 0 || text.empty())
        return text;

    const std::vector<std::string> raws = RawSentences(text);

    // One sentence is the whole answer, and the whole answer is spared on
    // purpose for direct address. Only a tail riding along behind something new
    // is in scope here.
    if (raws.size() < 2)
        return text;

    std::lock_guard<std::mutex> lock(g_mutex);
    const TimePoint now = Clock::now();

    auto botIt = g_bots.find(botGuid.GetRawValue());
    if (botIt == g_bots.end())
        return text;

    std::string kept;
    bool        dropped = false;

    for (const std::string& raw : raws)
    {
        const std::string norm = NormalizeText(raw);
        bool              repeat = false;

        if (!norm.empty() &&
            Tokenize(norm, false).size() >= size_t(g_SentenceRepeatMinWords))
        {
            for (const Utterance& u : botIt->second.history)
            {
                if (SecondsSince(u.when, now) > double(g_RepetitionWindowSeconds))
                    continue;
                if (std::find(u.sentences.begin(), u.sentences.end(), norm) != u.sentences.end())
                {
                    repeat = true;
                    break;
                }
            }
        }

        if (repeat)
        {
            dropped = true;
            ++g_stats.blockedRepetition;
            continue;
        }

        if (!kept.empty())
            kept.push_back(' ');
        kept += raw;
    }

    if (!dropped)
        return text;

    // Everything was a repeat: the bot is being asked the same thing again, and
    // the same answer is the right one. Silence would read as a fault.
    if (NormalizeText(kept).empty())
        return text;

    return kept;
}

// --- what was just said here ---------------------------------------------

void Governor_NoteScopeLine(const std::string& scopeKey, const std::string& speakerName,
                            const std::string& text)
{
    if (scopeKey.empty() || speakerName.empty() || text.empty())
        return;

    SpokenLine l;
    l.speaker = speakerName;
    l.text    = text;
    l.when    = Clock::now();

    std::lock_guard<std::mutex> lock(g_mutex);

    ScopeState& scope = g_scopes[scopeKey];
    scope.lines.push_back(std::move(l));

    while (scope.lines.size() > kMaxScopeLines)
        scope.lines.pop_front();
}

std::string Governor_RecentLines(const std::string& scopeKey, uint32_t maxLines,
                                 uint32_t skipMostRecent)
{
    if (maxLines == 0)
        return "";

    std::lock_guard<std::mutex> lock(g_mutex);
    const TimePoint now = Clock::now();

    auto scopeIt = g_scopes.find(scopeKey);
    if (scopeIt == g_scopes.end())
        return "";

    const std::deque<SpokenLine>& lines = scopeIt->second.lines;
    if (lines.size() <= size_t(skipMostRecent))
        return "";

    const size_t last  = lines.size() - size_t(skipMostRecent);   // one past the end
    const size_t first = last > size_t(maxLines) ? last - size_t(maxLines) : 0;

    std::string out;
    for (size_t i = first; i < last; ++i)
    {
        // An hour-old line is not what a follow-up is following.
        if (SecondsSince(lines[i].when, now) > double(g_RepetitionWindowSeconds))
            continue;

        out += lines[i].speaker;
        out += ": ";
        out += lines[i].text;
        out.push_back('\n');
    }

    return out;
}

// --- per-feature debounces -----------------------------------------------

bool Governor_TryConsumeEmoteReaction(ObjectGuid botGuid, ObjectGuid playerGuid)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    const TimePoint now = Clock::now();

    BotState& bot = g_bots[botGuid.GetRawValue()];
    TimePoint& last = bot.emoteReactions[playerGuid.GetRawValue()];

    if (SecondsSince(last, now) < double(g_EmoteReactionCooldownSeconds))
        return false;

    last = now;
    return true;
}

bool Governor_TryConsumeEventCooldown(ObjectGuid botGuid)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    const TimePoint now = Clock::now();

    BotState& bot = g_bots[botGuid.GetRawValue()];
    if (SecondsSince(bot.lastEvent, now) < double(g_EventCooldownTime))
        return false;

    bot.lastEvent = now;
    return true;
}

// --- maintenance ----------------------------------------------------------

void Governor_OnPlayerLogout(ObjectGuid guid)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    g_bots.erase(guid.GetRawValue());

    // Also drop this player from every bot's emote debounce table and from
    // their open conversations -- a fresh login starts a fresh exchange.
    const uint64_t raw = guid.GetRawValue();
    for (auto& [botGuid, state] : g_bots)
    {
        state.emoteReactions.erase(raw);
        state.conversations.erase(raw);
    }

    // And from every thread: both as the person holding one, and as the bot
    // being held onto. A bot that logs out must not keep the thread and leave
    // the next unnamed follow-up going to nobody.
    for (auto& [key, scope] : g_scopes)
    {
        scope.holders.erase(raw);

        for (auto it = scope.holders.begin(); it != scope.holders.end(); )
            it = (it->second.botGuid == raw) ? scope.holders.erase(it) : std::next(it);
    }
}

void Governor_Update()
{
    std::lock_guard<std::mutex> lock(g_mutex);
    const TimePoint now = Clock::now();

    TrimWindow(g_globalSends, now, 60.0);
    g_stats.sendsLastMinute = static_cast<uint32_t>(g_globalSends.size());

    const double staleAfter = std::max<double>(600.0, double(g_RepetitionWindowSeconds) * 2.0);

    for (auto it = g_scopes.begin(); it != g_scopes.end(); )
    {
        ScopeState& s = it->second;
        TrimWindow(s.sendTimes, now, 60.0);

        while (!s.history.empty() &&
               SecondsSince(s.history.front().when, now) > double(g_RepetitionWindowSeconds))
            s.history.pop_front();

        for (auto hit = s.holders.begin(); hit != s.holders.end(); )
            hit = (SecondsSince(hit->second.lastLineAt, now) > HolderWindowFor(hit->second))
                      ? s.holders.erase(hit)
                      : std::next(hit);

        // An expired quiet window is dead weight, and a scope holding one must
        // not be pruned while it still has force -- dropping it would silently
        // re-open the exchange it just ended.
        const bool quietLive = s.staleUntil.time_since_epoch().count() != 0 &&
                               now < s.staleUntil;
        if (!quietLive)
        {
            s.staleUntil = TimePoint{};
            s.staleHits  = 0;
        }

        const bool idle = s.history.empty() && s.sendTimes.empty() &&
                          s.holders.empty() && !quietLive &&
                          SecondsSince(s.lastHuman, now) > staleAfter &&
                          SecondsSince(s.lastSend, now)  > staleAfter;

        it = idle ? g_scopes.erase(it) : std::next(it);
    }

    for (auto& [guid, bot] : g_bots)
    {
        while (!bot.history.empty() &&
               SecondsSince(bot.history.front().when, now) > double(g_RepetitionWindowSeconds))
            bot.history.pop_front();

        for (auto it = bot.emoteReactions.begin(); it != bot.emoteReactions.end(); )
            it = (SecondsSince(it->second, now) > staleAfter)
                     ? bot.emoteReactions.erase(it)
                     : std::next(it);

        for (auto pit = bot.conversations.begin(); pit != bot.conversations.end(); )
        {
            for (auto sit = pit->second.begin(); sit != pit->second.end(); )
                sit = (SecondsSince(sit->second, now) > double(g_ConversationWindowSeconds))
                          ? pit->second.erase(sit)
                          : std::next(sit);

            pit = pit->second.empty() ? bot.conversations.erase(pit) : std::next(pit);
        }
    }

    g_stats.trackedBots   = static_cast<uint32_t>(g_bots.size());
    g_stats.trackedScopes = static_cast<uint32_t>(g_scopes.size());
}

void Governor_Reset()
{
    std::lock_guard<std::mutex> lock(g_mutex);
    g_bots.clear();
    g_scopes.clear();
    g_globalSends.clear();
    g_stats = GovernorStats{};
}

GovernorStats Governor_GetStats()
{
    std::lock_guard<std::mutex> lock(g_mutex);
    GovernorStats out = g_stats;
    out.trackedBots   = static_cast<uint32_t>(g_bots.size());
    out.trackedScopes = static_cast<uint32_t>(g_scopes.size());
    out.sendsLastMinute = static_cast<uint32_t>(g_globalSends.size());
    return out;
}
