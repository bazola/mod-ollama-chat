#include "mod-ollama-chat_conversation.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat_world.h"

#include "Log.h"
#include "MotionMaster.h"
#include "ObjectAccessor.h"
#include "Player.h"

#include "PlayerbotAI.h"
#include "PlayerbotMgr.h"

#include <chrono>
#include <numbers>
#include <unordered_map>

namespace
{
    using Clock = std::chrono::steady_clock;

    struct Engagement
    {
        uint64_t          personGuid = 0;
        Clock::time_point until;
        bool holding = false;
    };

    // Both directions, so either side is found without a scan. World thread
    // only, so unlocked.
    std::unordered_map<uint64_t, Engagement> g_byBot;
    std::unordered_map<uint64_t, uint64_t>   g_byPerson;   // person -> bot

    // Long enough that the AI never gets a tick between two of ours; short
    // enough that a released bot is back to itself within a second.
    constexpr uint32_t HOLD_AI_MS    = 1500;
    constexpr uint32_t HOLD_REFRESH_MS = 500;

    uint64_t Raw(Player* p) { return p->GetGUID().GetRawValue(); }

    Player* Find(uint64_t raw) { return ObjectAccessor::FindConnectedPlayer(ObjectGuid(raw)); }

    // Companions keep following their owner or group. A self-master is not
    // another player owning the bot, so it does not prevent a conversation.
    bool IsCompanion(Player* bot)
    {
        if (bot->GetGroup())
            return true;
        PlayerbotAI* botAI = PlayerbotsMgr::instance().GetPlayerbotAI(bot);
        Player* master = botAI ? botAI->GetMaster() : nullptr;
        return master && master != bot;
    }

    // Recovery and travel take precedence over a conversation. Even damage
    // outside combat must leave the AI free to heal, eat or escape.
    bool MayHold(Player* bot)
    {
        return bot->IsAlive() && !bot->IsInCombat() && bot->GetHealth() == bot->GetMaxHealth() &&
               !bot->IsInFlight() && !bot->GetTransport();
    }

    void Face(Player* bot, Player* person)
    {
        if (bot->HasUnitState(UNIT_STATE_CASTING) || bot->HasInArc(std::numbers::pi_v<float> / 6.0f, person))
            return;
        bot->SetFacingToObject(person);
    }

    void HoldStill(Player* bot, Player* person)
    {
        PlayerbotAI* botAI = PlayerbotsMgr::instance().GetPlayerbotAI(bot);
        if (!botAI)
            return;

        botAI->SetNextCheckDelay(HOLD_AI_MS);

        if (bot->isMoving())
        {
            bot->StopMoving();
            bot->GetMotionMaster()->Clear();
            bot->GetMotionMaster()->MoveIdle();
        }

        Face(bot, person);
    }

    void Release(uint64_t botGuid, const char* why)
    {
        auto it = g_byBot.find(botGuid);
        if (it == g_byBot.end())
            return;

        bool const wasHolding = it->second.holding;
        auto p = g_byPerson.find(it->second.personGuid);
        if (p != g_byPerson.end() && p->second == botGuid)
            g_byPerson.erase(p);
        g_byBot.erase(it);

        // Hand the bot back at once rather than when the last hold runs out:
        // if it is letting go because a fight started, it has to act now.
        if (Player* bot = Find(botGuid))
        {
            if (wasHolding)
                if (PlayerbotAI* botAI = PlayerbotsMgr::instance().GetPlayerbotAI(bot))
                    botAI->SetNextCheckDelay(0);

            if (g_DebugEnabled)
                LOG_INFO("module.ollamachat", "[Ollama Chat] {} is done talking ({}).", bot->GetName(), why);
        }
    }

    // Why a held conversation should end now, or nullptr to keep it.
    const char* EndReason(Player* bot, Player* person, const Engagement& e, Clock::time_point now)
    {
        if (!bot || !bot->IsInWorld())
            return "bot gone";
        if (!person || !person->IsInWorld())
            return "person gone";
        if (!bot->IsAlive())
            return "bot died";
        if (bot->GetHealth() < bot->GetMaxHealth())
            return "recovering";
        if (bot->IsInCombat() || person->IsInCombat())
            return "combat";
        if (!person->IsAlive())
            return "person died";
        if (bot->IsInFlight() || bot->GetTransport())
            return "travelling";
        if (!bot->IsWithinDistInMap(person, g_ConversationMaxDistance))
            return "out of range";
        if (IsCompanion(bot))
            return "companion";
        if (now >= e.until)
            return "silence";
        return nullptr;
    }
}

void Conversation_Engage(Player* bot, Player* person)
{
    if (!g_Enable || !g_ConversationEnable || !bot || !person || bot == person)
        return;
    if (!OllamaIsRealPlayer(person) || OllamaIsRealPlayer(bot))
        return;
    if (!PlayerbotsMgr::instance().GetPlayerbotAI(bot))
        return;
    const uint64_t botGuid    = Raw(bot);
    const uint64_t personGuid = Raw(person);

    // One partner each way: whoever this person was talking to before, and
    // whoever this bot was talking to before, are let go.
    auto previous = g_byPerson.find(personGuid);
    if (previous != g_byPerson.end() && previous->second != botGuid)
        Release(previous->second, "they turned to someone else");

    // A reply has already been accepted for this addressee. Switching to a
    // companion or recovering bot still ends the person's previous hold.
    if (!MayHold(bot) || !person->IsAlive() || person->IsInCombat() || IsCompanion(bot) ||
        !bot->IsWithinDistInMap(person, g_ConversationMaxDistance))
        return;

    auto current = g_byBot.find(botGuid);
    if (current != g_byBot.end() && current->second.personGuid != personGuid)
        Release(botGuid, "someone else spoke to it");

    const bool fresh = g_byBot.find(botGuid) == g_byBot.end();

    Engagement& e = g_byBot[botGuid];
    e.personGuid = personGuid;
    e.until      = Clock::now() + std::chrono::seconds(g_ConversationHoldSeconds);
    g_byPerson[personGuid] = botGuid;

    if (g_ConversationHoldStill)
    {
        HoldStill(bot, person);
        e.holding = true;
    }

    if (fresh && g_DebugEnabled)
        LOG_INFO("module.ollamachat", "[Ollama Chat] {} stops to talk with {}.", bot->GetName(), person->GetName());
}

bool Conversation_IsEngaged(Player* bot, Player* person)
{
    if (!g_ConversationEnable || !bot || !person)
        return false;
    auto it = g_byBot.find(Raw(bot));
    return it != g_byBot.end() && it->second.personGuid == Raw(person);
}

void Conversation_Refresh(Player* bot, Player* person)
{
    if (Conversation_IsEngaged(bot, person))
        Conversation_Engage(bot, person);
}

Player* Conversation_PartnerAmong(Player* person, const std::vector<Player*>& candidates)
{
    if (!g_ConversationEnable || !person)
        return nullptr;
    auto it = g_byPerson.find(Raw(person));
    if (it == g_byPerson.end())
        return nullptr;
    for (Player* bot : candidates)
        if (bot && Raw(bot) == it->second)
            return bot;
    return nullptr;
}

void Conversation_Update(uint32_t diff)
{
    static uint32_t timer = 0;
    timer += diff;
    const bool refreshHold = timer >= HOLD_REFRESH_MS;
    if (refreshHold)
        timer = 0;

    if (g_byBot.empty())
        return;

    // Switched off with a reload: let everyone go.
    if (!g_Enable || !g_ConversationEnable)
    {
        std::vector<uint64_t> all;
        for (const auto& [botGuid, e] : g_byBot)
            all.push_back(botGuid);
        for (uint64_t botGuid : all)
            Release(botGuid, "conversation mode off");
        return;
    }

    // Check safety on every world update; only reapplying the AI pause is
    // throttled. A bot taking damage must not wait for the next hold refresh.
    const Clock::time_point now = Clock::now();
    std::vector<std::pair<uint64_t, const char*>> ending;

    for (auto& [botGuid, e] : g_byBot)
    {
        Player* bot    = Find(botGuid);
        Player* person = Find(e.personGuid);
        if (const char* why = EndReason(bot, person, e, now))
        {
            ending.emplace_back(botGuid, why);
            continue;
        }
        if (g_ConversationHoldStill && refreshHold)
        {
            HoldStill(bot, person);
            e.holding = true;
        }
        else if (!g_ConversationHoldStill && e.holding)
        {
            if (PlayerbotAI* botAI = PlayerbotsMgr::instance().GetPlayerbotAI(bot))
                botAI->SetNextCheckDelay(0);
            e.holding = false;
        }
    }

    for (const auto& [botGuid, why] : ending)
        Release(botGuid, why);
}

uint32_t Conversation_Count()
{
    return static_cast<uint32_t>(g_byBot.size());
}

bool Conversation_HasPartner(Player* bot)
{
    return g_Enable && g_ConversationEnable && bot && g_byBot.find(Raw(bot)) != g_byBot.end();
}
