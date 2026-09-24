#include "mod-ollama-chat_response.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat_expression.h"
#include "mod-ollama-chat_text.h"

#include <algorithm>
#include <cctype>
#include <exception>
#include <string>
#include <vector>

// Trim, IsSentenceEnd, CountWords and the three length guards moved to mod-ollama-chat_text.cpp, which
// carries no project headers and so can be compiled and tested on its own (plans/35 §6). Everything left
// in this file needs the config PODs or expression.h behind it.
using OllamaText::IsSpace;
using OllamaText::Trim;

namespace
{
    // Decode one UTF-8 sequence at position i. Advances i past it.
    // Returns the codepoint, or 0xFFFD for malformed input (advancing by 1).
    uint32_t DecodeUtf8(const std::string& s, size_t& i, size_t& seqLen)
    {
        unsigned char c = static_cast<unsigned char>(s[i]);
        seqLen = 1;

        if (c <= 0x7F)
        {
            ++i;
            return c;
        }

        auto cont = [&](size_t off) -> bool
        {
            return i + off < s.size() &&
                   (static_cast<unsigned char>(s[i + off]) & 0xC0) == 0x80;
        };

        if ((c & 0xE0) == 0xC0 && cont(1))
        {
            uint32_t cp = ((c & 0x1Fu) << 6) |
                          (static_cast<unsigned char>(s[i + 1]) & 0x3Fu);
            i += 2;
            seqLen = 2;
            return cp;
        }
        if ((c & 0xF0) == 0xE0 && cont(1) && cont(2))
        {
            uint32_t cp = ((c & 0x0Fu) << 12) |
                          ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 6) |
                          (static_cast<unsigned char>(s[i + 2]) & 0x3Fu);
            i += 3;
            seqLen = 3;
            return cp;
        }
        if ((c & 0xF8) == 0xF0 && cont(1) && cont(2) && cont(3))
        {
            uint32_t cp = ((c & 0x07u) << 18) |
                          ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 12) |
                          ((static_cast<unsigned char>(s[i + 2]) & 0x3Fu) << 6) |
                          (static_cast<unsigned char>(s[i + 3]) & 0x3Fu);
            i += 4;
            seqLen = 4;
            return cp;
        }

        ++i;
        return 0xFFFD;
    }

    void AppendUtf8(std::string& out, uint32_t cp)
    {
        if (cp <= 0x7F)
        {
            out.push_back(static_cast<char>(cp));
        }
        else if (cp <= 0x7FF)
        {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
        else if (cp <= 0xFFFF)
        {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
        // Anything above the BMP is decoration we drop anyway.
    }
}

// --------------------------------------------------------------------------

std::string StripThinkTags(const std::string& text)
{
    static const std::string openTag  = "<think>";
    static const std::string closeTag = "</think>";

    std::string result = text;

    // Repeatedly remove complete blocks.
    for (;;)
    {
        size_t open = result.find(openTag);
        if (open == std::string::npos)
            break;

        size_t close = result.find(closeTag, open);
        if (close == std::string::npos)
        {
            // Unclosed block: the model was cut off mid-reasoning. Keep the
            // prefix rather than discarding the whole reply, which is what the
            // old code did.
            result.erase(open);
            break;
        }

        result.erase(open, (close + closeTag.size()) - open);
    }

    // A stray closing tag with no opener means reasoning leaked in ahead of it.
    size_t orphan = result.find(closeTag);
    if (orphan != std::string::npos)
        result.erase(0, orphan + closeTag.size());

    return Trim(result);
}

std::string UnwrapQuotedReply(const std::string& text)
{
    std::string s = Trim(text);
    if (s.size() < 2)
        return s;

    const char first = s.front();
    const char last  = s.back();

    const bool doubleQuoted = (first == '"' && last == '"');
    const bool singleQuoted = (first == '\'' && last == '\'');

    if (!doubleQuoted && !singleQuoted)
        return s;

    // Only unwrap when the quotes actually bracket the whole line -- if the
    // same quote character appears in between, this is dialogue, not wrapping.
    std::string inner = s.substr(1, s.size() - 2);
    if (inner.find(first) != std::string::npos)
        return s;

    return Trim(inner);
}

std::string StripSpeakerPrefix(const std::string& text, const std::string& botName)
{
    std::string s = Trim(text);
    if (s.empty())
        return s;

    auto lower = [](std::string v)
    {
        for (char& c : v)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return v;
    };

    // "<Name> msg" and "[Name] msg"
    if (s.front() == '<' || s.front() == '[')
    {
        const char closer = (s.front() == '<') ? '>' : ']';
        size_t end = s.find(closer);
        if (end != std::string::npos && end <= botName.size() + 2)
        {
            std::string inner = Trim(s.substr(1, end - 1));
            if (!botName.empty() && lower(inner) == lower(botName))
                return Trim(s.substr(end + 1));
        }
    }

    // "Name: msg" -- only strip when the name matches this bot, so a reply
    // that legitimately opens with "Warning: ..." survives.
    if (!botName.empty() && s.size() > botName.size())
    {
        if (lower(s.substr(0, botName.size())) == lower(botName))
        {
            size_t p = botName.size();
            while (p < s.size() && IsSpace(static_cast<unsigned char>(s[p])))
                ++p;
            if (p < s.size() && (s[p] == ':' || s[p] == '-'))
                return Trim(s.substr(p + 1));
        }
    }

    return s;
}

std::string StripPersonaSheet(const std::string& text, const std::string& botName)
{
    // The ambient template's own opening, spoken aloud. MEASURED 2026-09-24 (plans/50 §4): 68 lines, 1.52%
    // of every bot line that day, 54 of them in the literal second person of
    // RandomChatterPromptTemplate -- "You are Sylrela, a female night elf druid of the Alliance, living in
    // Azeroth. You stand in Stormwind City on the Eastern Kingdoms." One reached General chat reworded into
    // the third person. They are NOT truncated generations: median 19 words, correct punctuation, terminal
    // full stops. The model is treating its instructions as the line to say.
    //
    // Conservative in exactly the way StripSpeakerPrefix above is, and for the same reason: it acts only on
    // LEADING sentences and only when they name THIS bot, so an ordinary line that happens to open with
    // "You are late" or to mention somebody else survives untouched.
    //
    // It removes the sheet SENTENCE rather than the line, because 38 of those 68 carried real speech behind
    // the sheet; dropping whole lines would cost more than the echo does. A line that is nothing but sheet
    // (30 of 68) correctly reduces to empty, and the caller then skips it.
    if (!g_ResponseStripPersonaSheet || botName.empty())
        return text;

    auto lower = [](std::string v)
    {
        for (char& c : v)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return v;
    };
    auto has = [](const std::string& hay, const char* needle)
    {
        return hay.find(needle) != std::string::npos;
    };
    auto opens = [](const std::string& hay, const std::string& needle)
    {
        return hay.compare(0, needle.size(), needle) == 0;
    };

    // The template's own words, not a guess at what a persona looks like.
    static const char* const MARKERS[] = {
        "of the alliance", "of the horde", "living in azeroth",
        "on kalimdor", "on eastern kingdoms", "on the eastern kingdoms",
    };
    // Only ever considered AFTER a sheet opener has matched, so these cannot eat ordinary speech.
    static const char* const CONTINUATIONS[] = {
        "you are in", "you are ", "you stand", "you walk", "you speak", "you remember", "you carry",
        "you live", "i live in", "i am in", "she is in", "he is in", "they are in",
    };

    std::string s = Trim(text);
    const std::string name = lower(botName);
    int dropped = 0;

    // The model shortens names. MEASURED: a bot called Nedlaess was handed back "You are Ned, a night elf
    // rogue of the Alliance, living in Azeroth", which a whole-name test misses. So the word after the
    // opener need only be a PREFIX of this bot's name, three characters or more -- which still refuses
    // "You are late, Aldric", because "late" is no prefix of "aldric".
    auto namesThisBot = [&](const std::string& sent, const std::string& lead)
    {
        if (!opens(sent, lead))
            return false;
        size_t i = lead.size();
        std::string word;
        while (i < sent.size() &&
               (std::isalpha(static_cast<unsigned char>(sent[i])) || sent[i] == '\''))
            word.push_back(sent[i++]);
        return word.size() >= 3 && word.size() <= name.size() &&
               name.compare(0, word.size(), word) == 0;
    };

    while (!s.empty() && dropped < 3)
    {
        size_t end = 0;
        while (end < s.size() && s[end] != '.' && s[end] != '!' && s[end] != '?')
            ++end;
        if (end >= s.size())
            break;                       // no sentence break at all: leave the line alone

        const std::string sentence = lower(s.substr(0, end));

        bool marker = false;
        for (const char* m : MARKERS)
            marker = marker || has(sentence, m);

        // "You are <name>" / "I am <name>" needs nothing else: no line spoken in character opens by telling
        // itself who it is.
        const bool opener = namesThisBot(sentence, "you are ") || namesThisBot(sentence, "i am ");
        // "<name> stands tall, ... a warlock of the Alliance" -- self-narration in the third person. This one
        // does need a marker, because a line may legitimately begin with a name.
        const bool selfNarration = opens(sentence, name) && marker;

        bool continuation = false;
        if (dropped > 0)
            for (const char* c : CONTINUATIONS)
                continuation = continuation || opens(sentence, c);

        if (!opener && !selfNarration && !continuation)
            break;

        s = Trim(s.substr(end + 1));
        ++dropped;
    }

    return s;
}

std::string CollapseWhitespace(const std::string& text)
{
    std::string out;
    out.reserve(text.size());

    bool pendingSpace = false;
    for (char ch : text)
    {
        if (IsSpace(static_cast<unsigned char>(ch)))
        {
            pendingSpace = !out.empty();
            continue;
        }
        if (pendingSpace)
        {
            out.push_back(' ');
            pendingSpace = false;
        }
        out.push_back(ch);
    }

    return out;
}

std::string StripMarkdown(const std::string& text)
{
    std::string s = text;

    // Fenced code blocks first, so their contents don't get partially eaten.
    for (;;)
    {
        size_t open = s.find("```");
        if (open == std::string::npos)
            break;
        size_t close = s.find("```", open + 3);
        if (close == std::string::npos)
        {
            s.erase(open);
            break;
        }
        s.erase(open, (close + 3) - open);
    }

    std::string out;
    out.reserve(s.size());

    bool atLineStart = true;
    for (size_t i = 0; i < s.size(); ++i)
    {
        const char c = s[i];

        if (atLineStart)
        {
            // Leading heading markers, blockquotes and bullets.
            if (c == '#' || c == '>')
                continue;
            if ((c == '-' || c == '*' || c == '+') &&
                i + 1 < s.size() && s[i + 1] == ' ')
            {
                ++i; // also swallow the space
                continue;
            }
            if (!IsSpace(static_cast<unsigned char>(c)))
                atLineStart = false;
        }

        if (c == '\n')
        {
            atLineStart = true;
            out.push_back(c);
            continue;
        }

        // Emphasis and inline code markers.
        if (c == '*' || c == '_' || c == '`' || c == '~')
            continue;

        out.push_back(c);
    }

    return out;
}

std::string StripDecorativeUnicode(const std::string& text)
{
    std::string out;
    out.reserve(text.size());

    size_t i = 0;
    while (i < text.size())
    {
        size_t seqLen = 1;
        uint32_t cp = DecodeUtf8(text, i, seqLen);

        // Fold smart punctuation the model likes to emit down to ASCII.
        switch (cp)
        {
            case 0x2018: case 0x2019: case 0x201B:
                out.push_back('\'');
                continue;
            case 0x201C: case 0x201D: case 0x201F:
                out.push_back('"');
                continue;
            case 0x2013: case 0x2014: case 0x2212:
                out.push_back('-');
                continue;
            case 0x2026:
                out.append("...");
                continue;
            case 0x00A0: case 0x2007: case 0x202F:
                out.push_back(' ');
                continue;
            default:
                break;
        }

        if (cp == 0xFFFD)
            continue;

        // Keep ASCII and the Latin-1 / Latin Extended-A range so accented
        // names survive; drop arrows, dingbats, emoji and CJK.
        if (cp <= 0x7F || (cp >= 0xA0 && cp <= 0x24F))
        {
            AppendUtf8(out, cp);
            continue;
        }

        // Everything else is decoration the 3.3.5 client renders as boxes.
    }

    return out;
}

uint32_t TakeWordCap(std::string& text)
{
    const size_t at = text.find_last_of('@');
    if (at == std::string::npos || at + 1 >= text.size())
        return 0;
    for (size_t i = at + 1; i < text.size(); ++i)
        if (!std::isdigit(static_cast<unsigned char>(text[i])))
            return 0;
    uint32_t cap = 0;
    try { cap = static_cast<uint32_t>(std::stoul(text.substr(at + 1))); }
    catch (const std::exception&) { return 0; }
    text = Trim(text.substr(0, at));
    return cap;
}

// --------------------------------------------------------------------------

std::string ProcessLlmResponse(const std::string& raw,
                               const std::string& botName,
                               uint32_t* outEmoteId)
{
    if (outEmoteId)
        *outEmoteId = 0;

    if (raw.empty())
        return "";

    std::string s = raw;

    s = StripThinkTags(s);
    if (s.empty())
        return "";

    // Before StripMarkdown, not after. StripMarkdown removes every '*', so by
    // the time this used to run the "*waves*" gesture form could never match --
    // only [emote:name] and /wave survived. ExtractEmoteTag leaves the words of
    // a non-gesture asterisk span in place, so emphasis still reads correctly
    // once StripMarkdown takes the markers off below.
    {
        uint32_t emote = ExtractEmoteTag(s);
        if (outEmoteId)
            *outEmoteId = emote;
    }

    if (g_ResponseStripMarkdown)
        s = StripMarkdown(s);

    s = CollapseWhitespace(s);
    s = UnwrapQuotedReply(s);
    s = StripSpeakerPrefix(s, botName);
    // After the speaker prefix, so "Sylrela: You are Sylrela, a night elf druid..." loses both halves.
    s = StripPersonaSheet(s, botName);

    if (g_ResponseStripDecorativeUnicode)
        s = StripDecorativeUnicode(s);

    // Unwrap again: stripping a prefix or a tag can expose wrapping quotes
    // that were not on the outside before.
    s = UnwrapQuotedReply(s);
    s = CollapseWhitespace(s);
    s = DropUnfinishedTail(s);
    s = ClampReplyLength(s, g_MaxReplyLength);

    return Trim(s);
}
