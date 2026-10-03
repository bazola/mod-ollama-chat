#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat_handler.h"
#include "mod-ollama-chat_random.h"
#include "mod-ollama-chat_events.h"
#include "mod-ollama-chat_command.h"
#include "mod-ollama-chat_expression.h"
#include "mod-ollama-chat_rag.h"
#include "mod-ollama-chat_director.h"
#include "Log.h"

void Addmod_ollama_chatScripts()
{
    LOG_INFO("server.loading", "[Ollama Chat] Registering mod-ollama-chat scripts.");
    new OllamaChatConfigWorldScript();
    new PlayerBotChatHandler();
    new OllamaChatMaintenance();
    new OllamaBotRandomChatter();

    LOG_INFO("server.loading", "[Ollama Chat] Registering mod-ollama-chat events.");
    new ChatOnKill();
    new ChatOnLoot();
    new ChatOnDeath();
    new ChatOnQuest();
    new ChatOnLearn();
    new ChatOnDuel();
    new ChatOnLevelUp();
    new ChatOnAchievement();
    new ChatOnGameObjectUse();

    // Guild events. ChatOnGuildMemberChange used to exist but was never
    // registered here, and its hooks were not AzerothCore hooks in any case.
    new ChatOnGuild();
    new ChatOnGuildLogin();

    // Bots react when a player emotes at them.
    new ChatOnEmote();

    // The Dungeon Master's scene channel (plan 62).
    new OllamaDirectorWorldScript();
    new OllamaDirectorPlayerScript();
    new OllamaDirectorMapScript();

    new OllamaChatConfigCommand();
}
