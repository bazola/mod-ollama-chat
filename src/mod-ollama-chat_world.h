#ifndef MOD_OLLAMA_CHAT_WORLD_H
#define MOD_OLLAMA_CHAT_WORLD_H

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

class Channel;
class Player;

// --------------------------------------------------------------------------
// One pass over the online player list, reused for the whole of a message or
// a chatter tick.
//
// The module repeatedly asked the same three questions -- "is a real player
// near this bot", "does this bot's guild have a real player online", "is a
// real player in this bot's zone and faction" -- and each one walked
// ObjectAccessor::GetPlayers() from scratch. Asked once per candidate bot,
// inside a loop that was itself over every online player, that is quadratic in
// online characters on every single chat message.
//
// Build one of these once, then answer all three in O(real players), which on
// a bot-heavy realm is a very small number.
//
// WORLD THREAD ONLY, and do not hold one across ticks: the Player pointers are
// only guaranteed valid for the tick that built it.
// --------------------------------------------------------------------------
// Authoritative "is this a bot" test.
//
// PlayerbotsMgr::GetPlayerbotAI() only answers correctly once the AI has been
// attached, and for a freshly logged-in bot that happens inside mod-playerbots'
// own PLAYERHOOK_ON_LOGIN handler. Script hook order between modules is not
// defined, so during login a bot can read as a real player.
//
// That is not theoretical: it is why an all-bot realm generated guild login
// chatter for an audience of nobody. The bot's own login satisfied the
// "is a real player online in this guild" gate, the module spent an LLM round
// trip on a reply, and by the time it was delivered the AI had attached and
// the same check correctly said there was no one to talk to.
//
// WorldSession::_isBot is set in the session constructor, so it is correct from
// the first moment the Player exists. mod-playerbots' own login hook uses it
// for exactly this reason. The AI lookup is kept as a fallback so a bot driven
// by some other mechanism still reads as a bot.
bool OllamaIsBotPlayer(Player* player);

inline bool OllamaIsRealPlayer(Player* player)
{
    return player && !OllamaIsBotPlayer(player);
}

// True when `speaker` is `bot`'s master and mod-playerbots will run `msg` as an
// order to it (local patch, custom wow plan 21 P7).
//
// A bot-control addon such as CleanBot sends orders by the dozen ("co ?",
// "maintenance", "s gray"). A bot answering one in character is answering
// nothing, and BlacklistCommands can only guess at every addon's words.
// `whisper` says the order came as a whisper, which mod-playerbots reads a
// little differently ("queue" is not an order there).
//
// Mirrors PlayerbotAI::HandleCommand and ExternalEventHelper::ParseChatCommand;
// keep it in step when mod-playerbots is rebased.
bool OllamaIsCommandFromMaster(Player* bot, Player* speaker, std::string const& msg, bool whisper);

// The continent a bot would actually name, for prompt use.
//
// Map::GetMapName() answers with the map's own name, and the Burning Crusade
// starting zones share map 530 with Outland in the game data -- so a Blood Elf
// standing in Ghostlands is told they are in Outland. The model notices: one
// reply was spent reasoning about the contradiction instead of answering it.
//
// Everything else GetMapName() returns is already right, instances included,
// so this only rewrites the handful of zones that are wrong.
std::string OllamaContinentName(Player* player);

// True when this player's party or raid contains at least one human. A group
// of nothing but bots is not an audience.
//
// Free rather than a snapshot method: it walks the group, not the online
// player list, so it needs no snapshot state and callers without one can use
// it.
bool OllamaGroupHasRealPlayer(Player* who);

struct OllamaWorldSnapshot
{
    std::vector<Player*>        realPlayers;            // non-bot, in world
    std::unordered_set<uint32_t> guildsWithRealPlayer;

    void Build();

    bool Empty() const { return realPlayers.empty(); }

    bool GuildHasRealPlayer(uint32_t guildId) const
    {
        return guildId != 0 && guildsWithRealPlayer.count(guildId) != 0;
    }

    // True when a real player is within `distance` of `who`, on the same map.
    bool RealPlayerWithin(Player* who, float distance) const;

    // True when a real player shares this bot's zone and faction. This is a
    // cheap PRE-FILTER, not an audience test: being in the zone does not mean
    // being in the channel. Use RealPlayerInChannel for the real answer.
    bool RealPlayerInZoneAndFaction(Player* who) const;

    // The actual audience tests. Nothing may be sent to the LLM for a
    // destination unless one of these says a human would see it -- an answer
    // nobody reads still costs a full generation.
    bool RealPlayerInChannel(Channel* channel) const;
};

#endif // MOD_OLLAMA_CHAT_WORLD_H
