#ifndef MOD_OLLAMA_CHAT_TEXT_H
#define MOD_OLLAMA_CHAT_TEXT_H

#include <cstddef>
#include <string>

// --------------------------------------------------------------------------
// The pure half of the response pipeline: string work with no config, no game
// objects and no project headers behind it.
//
// mod-ollama-chat_text.cpp defines these helpers and the three length guards
// declared in mod-ollama-chat_response.h (ClampReplyLength, ClampReplyWords,
// DropUnfinishedTail). Keeping them in their own translation unit is what lets
// them be compiled and tested on their own -- response.cpp pulls in config.h
// (-> ScriptMgr.h) and expression.h (-> ObjectGuid.h), and stubbing those out
// turns into rebuilding a stub AzerothCore (plans/35 §6).
//
// Namespaced deliberately: every module links statically into the one
// worldserver binary, so a global symbol named Trim or CountWords is an
// invitation to collide with another module's.
// --------------------------------------------------------------------------

namespace OllamaText
{
    inline bool IsSpace(unsigned char c)
    {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    }

    // Strip leading and trailing whitespace.
    std::string Trim(const std::string& s);

    // True when s[i] is a '.', '!' or '?' that actually ends a sentence: the next character is a space, a
    // closing quote, a ')' or the end of the text. Keeps "St. Alia" from counting mid-word.
    bool IsSentenceEnd(const std::string& s, size_t i);

    // Words in the half-open byte range [from, to).
    size_t CountWords(const std::string& s, size_t from, size_t to);
}

#endif // MOD_OLLAMA_CHAT_TEXT_H
