#ifndef MOD_OLLAMA_CHAT_DIRECTOR_H
#define MOD_OLLAMA_CHAT_DIRECTOR_H

#include "ScriptMgr.h"

// --------------------------------------------------------------------------
// The Dungeon Master's scene channel (custom wow plan 62).
//
// custom-wow services/dm/dm.py decides WHAT is said and for whom, early, and
// writes it to dm_line. This file decides WHEN, from positions only the
// worldserver has: it loads dm_line on its own short timer and, when a real
// player comes within sight of a dungeon's final boss, that boss yells the
// line once per instance. Nothing here calls a model or makes a query on the
// world thread; the loader runs on a detached thread like the regard loader.
//
// Delivery is recorded through mod-ledger (LedgerRecordScene, a weak symbol),
// never by writing back to dm_line, which has one writer: dm.py.
// --------------------------------------------------------------------------

// Loads dm_line on OllamaChat.Director.RefreshSeconds.
class OllamaDirectorWorldScript : public WorldScript
{
public:
    OllamaDirectorWorldScript();
    void OnUpdate(uint32 diff) override;
};

// The approach check: a real player in a dungeon, once a second.
class OllamaDirectorPlayerScript : public PlayerScript
{
public:
    OllamaDirectorPlayerScript();
    void OnPlayerUpdate(Player* player, uint32 diff) override;
    void OnPlayerLogout(Player* player) override;
};

// Forgets which bosses have spoken in an instance once it is gone, because
// instance ids are reused.
class OllamaDirectorMapScript : public AllMapScript
{
public:
    OllamaDirectorMapScript();
    void OnDestroyInstance(MapInstanced* mapInstanced, Map* map) override;
};

#endif // MOD_OLLAMA_CHAT_DIRECTOR_H
