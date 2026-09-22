// Offline tests for the pure text guards (plans/35 §6).
//
// Build and run, from the module root:
//   g++ -std=c++17 -Wall -Wextra -o /tmp/text_test apps/text-test/text_test.cpp src/mod-ollama-chat_text.cpp -Isrc && /tmp/text_test
//
// Two files, no stubs, no AzerothCore: that is the whole point of the split. src/mod-ollama-chat_text.cpp
// includes only mod-ollama-chat_response.h (<string>, <cstdint>) and mod-ollama-chat_text.h.

#include "mod-ollama-chat_response.h"
#include "mod-ollama-chat_text.h"

#include <cstdio>
#include <string>

static int g_failed = 0;
static int g_ran = 0;

static void check(const std::string& name, const std::string& got, const std::string& want)
{
    ++g_ran;
    if (got == want)
        return;
    ++g_failed;
    std::printf("FAIL %s\n  got  \"%s\"\n  want \"%s\"\n", name.c_str(), got.c_str(), want.c_str());
}

static void checkTrue(const std::string& name, bool ok, const std::string& detail)
{
    ++g_ran;
    if (ok)
        return;
    ++g_failed;
    std::printf("FAIL %s: %s\n", name.c_str(), detail.c_str());
}

static size_t words(const std::string& s)
{
    return OllamaText::CountWords(s, 0, s.size());
}

int main()
{
    // ---------------------------------------------------------------------
    // 1. The line that started this: 90 words, not one '.', '!' or '?' and not one comma, against @55.
    // Before the fix all three guards returned it unchanged (plans/35 §3).
    // ---------------------------------------------------------------------
    {
        std::string line;
        for (int i = 0; i < 90; ++i)
            line += (i ? " " : "") + std::string("ash");
        const std::string got = ClampReplyWords(line, 55);
        checkTrue("90 unpunctuated words: at or under the cap",
                  words(got) <= 55, "got " + std::to_string(words(got)) + " words");
        checkTrue("90 unpunctuated words: ends on a full stop",
                  !got.empty() && got.back() == '.', "got \"" + got + "\"");
        // Whole words only, so every token is still "ash". Do NOT test this with substring searches:
        // "ash" contains "as", and the first version of this check failed on correct output.
        std::string want;
        for (int i = 0; i < 55; ++i)
            want += (i ? " " : "") + std::string("ash");
        want += ".";
        check("90 unpunctuated words: exactly the cap, no mid-word cut", got, want);
    }

    // A realistic version of the same shape, so the cut point is readable in a failure.
    check("unpunctuated prose, cap 8",
          ClampReplyWords("the road out of Kharanos is long and cold and the wind never lets up at all", 8),
          "the road out of Kharanos is long and.");

    // ---------------------------------------------------------------------
    // 2. Under the cap: untouched.
    // ---------------------------------------------------------------------
    check("under the cap", ClampReplyWords("Cold morning.", 55), "Cold morning.");
    check("exactly at the cap", ClampReplyWords("one two three", 3), "one two three");

    // ---------------------------------------------------------------------
    // 3. maxWords == 0 means no cap at all (emote reactions, dispatch.cpp).
    // ---------------------------------------------------------------------
    {
        const std::string long_ = "a b c d e f g h i j k l m n o p q r s t u v w x y z";
        check("cap of zero is no cap", ClampReplyWords(long_, 0), long_);
    }

    // ---------------------------------------------------------------------
    // 4. REGRESSION GUARD (plans/35 §4): a sentence end just past the cap still yields the whole first
    // sentence. The fix must not touch this path -- bounding it is deliberately out of scope.
    // ---------------------------------------------------------------------
    check("sentence end just past the cap keeps the whole sentence",
          ClampReplyWords("the dead do not rest easy in these hills tonight. I keep watch.", 12),
          "the dead do not rest easy in these hills tonight.");

    // ---------------------------------------------------------------------
    // 5. Commas but no sentence end: cut at the last clause break inside the cap.
    // ---------------------------------------------------------------------
    check("clause break inside the cap",
          ClampReplyWords("the forge is cold, the ore is spent, and nobody has come up the road in days", 8),
          "the forge is cold, the ore is spent.");

    // ---------------------------------------------------------------------
    // 6. An ellipsis is NOT a sentence end mid-string, but a trailing "..." is (i + 1 >= size).
    // So this text has a sentence end and takes the §4 out-of-scope path: the whole first "sentence",
    // unclamped. Asserted as documented behaviour, not endorsed -- plans/35 §4 leaves it alone until
    // something actually shows up in the corpus.
    // ---------------------------------------------------------------------
    {
        const std::string ell = "the ash falls and the road is long and cold and we walk it...";
        check("trailing ellipsis: documented out-of-scope behaviour", ClampReplyWords(ell, 5), ell);
    }

    // ---------------------------------------------------------------------
    // 7. A single word longer than the cap is one word, so it is under the cap and never cut mid-word.
    // ---------------------------------------------------------------------
    check("single long word", ClampReplyWords("Thelsamarrrrrrrrrr", 5), "Thelsamarrrrrrrrrr");
    check("two words, cap of one, no mid-word cut", ClampReplyWords("stone mace", 1), "stone.");

    // ---------------------------------------------------------------------
    // 8. A closing quote or paren after the terminator is retained.
    // ---------------------------------------------------------------------
    check("closing quote retained",
          ClampReplyWords("\"hold the line.\" the rest can wait until morning light", 4),
          "\"hold the line.\"");
    check("closing paren retained",
          ClampReplyWords("(he is gone.) nobody speaks of it now in the hall", 4),
          "(he is gone.)");

    // ---------------------------------------------------------------------
    // The other two guards moved in the same commit and must behave exactly as before.
    // ---------------------------------------------------------------------
    check("DropUnfinishedTail drops an unfinished thought",
          DropUnfinishedTail("I keep watch. The wind is up and"),
          "I keep watch.");
    check("DropUnfinishedTail leaves a line with no complete sentence alone",
          DropUnfinishedTail("the wind is up and"),
          "the wind is up and");
    check("DropUnfinishedTail keeps a closing quote",
          DropUnfinishedTail("\"hold the line.\""),
          "\"hold the line.\"");

    check("ClampReplyLength cuts at a sentence in the last third",
          ClampReplyLength("Cold morning. The road is long and the wind never lets up at all out here.", 30),
          "Cold morning.");
    check("ClampReplyLength falls back to the last word boundary",
          ClampReplyLength("abcdefghij klmnopqrst uvwxyz", 26),
          "abcdefghij klmnopqrst");
    // Documented quirk, NOT changed here: the fallback is guarded by `lastSpace > head.size() / 2`, so
    // when the only space sits at or before the midpoint the byte cut stands and a word is split. It is
    // pre-existing behaviour and plans/35 §4 scopes the fix to ClampReplyWords alone; recorded so the
    // next person to read this file knows it was seen and left deliberately.
    check("ClampReplyLength: an early space leaves the mid-word cut standing",
          ClampReplyLength("abcdefghij klmnopqrst uvwxyz", 20),
          "abcdefghij klmnopqrs");
    check("ClampReplyLength under the limit", ClampReplyLength("short", 700), "short");
    check("ClampReplyLength zero means no limit", ClampReplyLength("short", 0), "short");

    // TakeWordCap is NOT in this build: it stayed in response.cpp with the config PODs.

    std::printf("%s: %d checks, %d failed\n", g_failed ? "FAILURES" : "ok", g_ran, g_failed);
    return g_failed ? 1 : 0;
}
