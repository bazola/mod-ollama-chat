#include "mod-ollama-chat_response.h"
#include "mod-ollama-chat_text.h"

#include <cctype>
#include <string>

// The pure text guards, split out of mod-ollama-chat_response.cpp so they can be built and tested without
// AzerothCore behind them (plans/35 §6). Behaviour is unchanged from that file except for the one fix in
// ClampReplyWords marked below.

namespace OllamaText
{
    std::string Trim(const std::string& s)
    {
        size_t start = 0;
        size_t end = s.size();
        while (start < end && IsSpace(static_cast<unsigned char>(s[start])))
            ++start;
        while (end > start && IsSpace(static_cast<unsigned char>(s[end - 1])))
            --end;
        return s.substr(start, end - start);
    }

    bool IsSentenceEnd(const std::string& s, size_t i)
    {
        // i indexes a '.', '!' or '?'; a sentence ends there if the next character is space, a closing
        // quote or the end of the text. Keeps "St. Alia" style abbreviations from counting mid-word.
        const char c = s[i];
        if (c != '.' && c != '!' && c != '?')
            return false;
        if (i + 1 >= s.size())
            return true;
        const char n = s[i + 1];
        return n == ' ' || n == '"' || n == '\'' || n == ')';
    }

    size_t CountWords(const std::string& s, size_t from, size_t to)
    {
        size_t words = 0;
        bool inWord = false;
        for (size_t i = from; i < to && i < s.size(); ++i)
        {
            const bool space = std::isspace(static_cast<unsigned char>(s[i])) != 0;
            if (!space && !inWord)
                ++words;
            inWord = !space;
        }
        return words;
    }
}

using OllamaText::CountWords;
using OllamaText::IsSentenceEnd;
using OllamaText::IsSpace;
using OllamaText::Trim;

std::string ClampReplyLength(const std::string& text, uint32_t maxLen)
{
    if (maxLen == 0 || text.size() <= maxLen)
        return text;

    // Back off to a UTF-8 boundary at or before maxLen.
    size_t cut = maxLen;
    while (cut > 0 && (static_cast<unsigned char>(text[cut]) & 0xC0) == 0x80)
        --cut;

    std::string head = text.substr(0, cut);

    // Prefer ending on a complete sentence if one lands in the last third.
    const size_t minSentence = head.size() > 40 ? head.size() * 2 / 3 : 0;
    size_t bestSentence = std::string::npos;
    for (size_t i = head.size(); i > minSentence; --i)
    {
        const char c = head[i - 1];
        if (c == '.' || c == '!' || c == '?')
        {
            bestSentence = i;
            break;
        }
    }
    if (bestSentence != std::string::npos)
        return Trim(head.substr(0, bestSentence));

    // Otherwise cut at the last word boundary so we never truncate mid-word.
    size_t lastSpace = head.find_last_of(' ');
    if (lastSpace != std::string::npos && lastSpace > head.size() / 2)
        head.erase(lastSpace);

    return Trim(head);
}

std::string ClampReplyWords(const std::string& text, uint32_t maxWords)
{
    if (maxWords == 0 || CountWords(text, 0, text.size()) <= maxWords)
        return text;

    // Whole sentences while they fit.
    size_t keep = 0;
    for (size_t i = 0; i < text.size(); ++i)
    {
        if (!IsSentenceEnd(text, i))
            continue;
        size_t end = i + 1;
        while (end < text.size() && (text[end] == '"' || text[end] == '\'' || text[end] == ')'))
            ++end;
        if (CountWords(text, 0, end) > maxWords)
            break;
        keep = end;
    }
    if (keep > 0)
        return Trim(text.substr(0, keep));

    // The first sentence alone overruns: end it at the last clause break inside the cap.
    size_t words = 0, capEnd = text.size();
    bool inWord = false;
    for (size_t i = 0; i < text.size(); ++i)
    {
        const bool space = std::isspace(static_cast<unsigned char>(text[i])) != 0;
        if (!space && !inWord && ++words > maxWords)
        {
            capEnd = i;
            break;
        }
        inWord = !space;
    }
    // Commas and semicolons only: after unicode folding a dash may sit inside a word ("stone-mace") as
    // easily as between clauses, and cutting there leaves half a word.
    const size_t brk = text.substr(0, capEnd).find_last_of(",;");
    if (brk == std::string::npos || CountWords(text, 0, brk) < 3)
    {
        // No clause to stop at: the whole first sentence is better than a broken one.
        for (size_t i = 0; i < text.size(); ++i)
            if (IsSentenceEnd(text, i))
                return Trim(text.substr(0, i + 1));

        // And when there is no sentence either, the rule above has nothing to protect: reaching this
        // point means the text carries no '.', '!' or '?' anywhere at all, which is how one 90-word
        // ambient line walked through three separate guards untouched (plans/35 §3.1). capEnd is already
        // the first byte of the word that would overrun, so cutting there keeps whole words; ending the
        // line lets it read as a finished thought rather than a severed one.
        std::string head = Trim(text.substr(0, capEnd));
        while (!head.empty() && (head.back() == ',' || head.back() == ';' || head.back() == '-' ||
                                 IsSpace(static_cast<unsigned char>(head.back()))))
            head.pop_back();
        return head.empty() ? text : head + ".";
    }
    std::string head = Trim(text.substr(0, brk));
    while (!head.empty() && (head.back() == '-' || head.back() == ' '))
        head.pop_back();
    return head.empty() ? text : head + ".";
}

std::string DropUnfinishedTail(const std::string& text)
{
    std::string s = Trim(text);
    if (s.empty())
        return s;
    char last = s.back();
    size_t tail = s.size();
    while (tail > 0 && (s[tail - 1] == '"' || s[tail - 1] == '\'' || s[tail - 1] == ')'))
        --tail;
    if (tail > 0)
        last = s[tail - 1];
    if (last == '.' || last == '!' || last == '?')
        return s;

    for (size_t i = s.size(); i-- > 0; )
    {
        if (IsSentenceEnd(s, i))
        {
            size_t end = i + 1;
            while (end < s.size() && (s[end] == '"' || s[end] == '\'' || s[end] == ')'))
                ++end;
            return Trim(s.substr(0, end));
        }
    }
    return s;
}
