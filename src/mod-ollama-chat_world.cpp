#include "mod-ollama-chat_world.h"

#include "AreaDefines.h"
#include "Channel.h"
#include "Map.h"
#include "Group.h"
#include "ObjectAccessor.h"
#include "Player.h"

#include "WorldSession.h"

#include "AiObjectContext.h"
#include "ChatTriggerContext.h"
#include "PlayerbotAI.h"
#include "PlayerbotAIConfig.h"
#include "PlayerbotMgr.h"

#include <cstring>
#include <set>

bool OllamaIsBotPlayer(Player* player)
{
    if (!player)
        return false;

    // Correct from session construction, unlike the AI lookup below.
    if (WorldSession* session = player->GetSession())
    {
        if (session->IsBot())
            return true;
    }

    PlayerbotAI* ai = PlayerbotsMgr::instance().GetPlayerbotAI(player);
    return ai && ai->IsBotAI();
}

namespace
{
    // The names mod-playerbots runs as chat commands. ChatTriggerContext fills its
    // creators in the constructor and needs no bot, so one throwaway instance gives
    // the whole set. The bot's own trigger context is not asked: looking a name up
    // there caches an empty entry for every word tried.
    std::unordered_set<std::string> const& ChatCommandNames()
    {
        static std::unordered_set<std::string> const names = []
        {
            ChatTriggerContext context;
            std::set<std::string> const keys = context.supports();
            return std::unordered_set<std::string>(keys.begin(), keys.end());
        }();
        return names;
    }

    std::string Trimmed(std::string const& text)
    {
        size_t const first = text.find_first_not_of(" \t\r\n");
        if (first == std::string::npos)
            return "";

        return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
    }

    bool StartsWith(std::string const& text, char const* prefix)
    {
        return text.rfind(prefix, 0) == 0;
    }

    // One order (no separator), read the way PlayerbotAI::HandleCommand and
    // ExternalEventHelper::ParseChatCommand read it.
    bool IsCommandPart(PlayerbotAI* ai, std::string text, bool whisper)
    {
        std::string const& prefix = sPlayerbotAIConfig.commandPrefix;
        if (!prefix.empty())
        {
            if (!StartsWith(text, prefix.c_str()))
                return false;

            text = text.substr(prefix.size());
        }

        // "#w ", "#p ", "#r ", "#a ", "#g " only choose where the answer comes back.
        if (text.size() >= 3 && text[0] == '#' && text[2] == ' ' && std::strchr("wprag", text[1]))
            text = text.substr(3);

        text = Trimmed(text);
        if (text.empty())
            return false;

        // "@tank follow", "@60 stay": an order addressed to some of the bots, this one or not.
        if (text[0] == '@')
            return true;

        if (StartsWith(text, "debug ") || text == "reset" || text == "logout")
            return true;

        // "do <action>" runs a named action. Only a real one: "do you remember" is speech.
        if ((text.size() > 2 && StartsWith(text, "d ")) || (text.size() > 3 && StartsWith(text, "do ")))
        {
            std::set<std::string> const actions = ai->GetAiObjectContext()->GetSupportedActions();
            return actions.count(text.substr(text.find(' ') + 1)) != 0;
        }

        if (!whisper && text.size() > 6 && StartsWith(text, "queue "))
            return IsCommandPart(ai, text.substr(6), whisper);

        // The whole text names a command, or a shorter head of it does ("pull my
        // target" -> "pull"). ParseChatCommand also takes an item link on its own as
        // a trade offer; that is left to speech, since people show each other things.
        std::unordered_set<std::string> const& names = ChatCommandNames();
        if (names.count(text))
            return true;

        for (size_t space = text.rfind(' '); space != std::string::npos && space != 0; space = text.rfind(' ', space - 1))
        {
            if (names.count(text.substr(0, space)))
                return true;
        }

        return false;
    }
}

bool OllamaIsCommandFromMaster(Player* bot, Player* speaker, std::string const& msg, bool whisper)
{
    if (!bot || !speaker || bot == speaker || msg.empty())
        return false;

    PlayerbotAI* ai = PlayerbotsMgr::instance().GetPlayerbotAI(bot);
    if (!ai || !ai->IsBotAI() || ai->GetMaster() != speaker)
        return false;

    std::string const& separator = sPlayerbotAIConfig.commandSeparator;
    if (separator.empty() || msg.find(separator) == std::string::npos)
        return IsCommandPart(ai, msg, whisper);

    // mod-playerbots runs every part; one order among them is enough.
    for (size_t start = 0;;)
    {
        size_t const end = msg.find(separator, start);
        if (IsCommandPart(ai, msg.substr(start, end == std::string::npos ? std::string::npos : end - start), whisper))
            return true;

        if (end == std::string::npos)
            return false;

        start = end + separator.size();
    }
}

std::string OllamaContinentName(Player* player)
{
    Map* map = player ? player->GetMap() : nullptr;
    if (!map)
        return "UnknownMap";

    // The Burning Crusade starting zones sit on Outland's map but belong to
    // the old continents in the fiction, which is what a character would say.
    if (map->GetId() == MAP_OUTLAND)
    {
        switch (player->GetZoneId())
        {
            case AREA_EVERSONG_WOODS:
            case AREA_GHOSTLANDS:
            case AREA_SILVERMOON_CITY:
            case AREA_ISLE_OF_QUEL_DANAS:
                return "Eastern Kingdoms";

            case AREA_AZUREMYST_ISLE:
            case AREA_BLOODMYST_ISLE:
            case AREA_THE_EXODAR:
                return "Kalimdor";

            default:
                break;
        }
    }

    return map->GetMapName();
}

void OllamaWorldSnapshot::Build()
{
    realPlayers.clear();
    guildsWithRealPlayer.clear();

    auto const& all = ObjectAccessor::GetPlayers();
    realPlayers.reserve(16);

    for (auto const& pair : all)
    {
        Player* player = pair.second;
        if (!player || !player->IsInWorld())
            continue;

        if (OllamaIsBotPlayer(player))
            continue;

        realPlayers.push_back(player);

        if (uint32_t guildId = player->GetGuildId())
            guildsWithRealPlayer.insert(guildId);
    }
}

bool OllamaWorldSnapshot::RealPlayerWithin(Player* who, float distance) const
{
    if (!who || distance <= 0.0f || !who->IsInWorld())
        return false;

    for (Player* player : realPlayers)
    {
        if (player == who)
            continue;
        if (player->GetMapId() != who->GetMapId())
            continue;
        if (who->GetDistance(player) <= distance)
            return true;
    }
    return false;
}

bool OllamaWorldSnapshot::RealPlayerInChannel(Channel* channel) const
{
    if (!channel)
        return false;

    for (Player* player : realPlayers)
        if (player->IsInChannel(channel))
            return true;

    return false;
}

bool OllamaGroupHasRealPlayer(Player* who)
{
    Group* group = who ? who->GetGroup() : nullptr;
    if (!group)
        return false;

    for (GroupReference* ref = group->GetFirstMember(); ref; ref = ref->next())
        if (OllamaIsRealPlayer(ref->GetSource()))
            return true;

    return false;
}

bool OllamaWorldSnapshot::RealPlayerInZoneAndFaction(Player* who) const
{
    if (!who)
        return false;

    for (Player* player : realPlayers)
    {
        if (player == who)
            continue;
        if (player->GetTeamId() == who->GetTeamId() &&
            player->GetZoneId() == who->GetZoneId())
            return true;
    }
    return false;
}
