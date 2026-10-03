#ifndef MOD_OLLAMA_CHAT_CONVERSATION_H
#define MOD_OLLAMA_CHAT_CONVERSATION_H

#include <cstdint>
#include <vector>

class Player;

// --------------------------------------------------------------------------
// Conversation mode.
//
// A person who speaks to a bot face to face holds its attention. Until now a
// bot answered and walked on with whatever it was doing: it could be twenty
// yards off by the time its reply landed, and the person's next line went to
// whichever bot happened to roll for it. The governor's conversation window
// only raised a reply chance.
//
// While a bot is engaged with someone:
//   - it stops, turns to them and holds still (its AI is paused and its
//     movement cleared), when OllamaChat.Conversation.HoldStill is on;
//   - their say and yell lines go to it and only it, unless they name
//     another bot;
//   - their lines count as direct address, so pacing does not swallow them.
//
// It lets go when they have said nothing for HoldSeconds, walk out of
// MaxDistance, change map, log out or die -- and at once when the bot is drawn
// into a fight, so nothing here ever leaves a bot standing still under attack.
// Grouped or mastered bots are never held, and only a bot at full health may
// be engaged. Safety is checked every world update; any damage ends the hold
// so recovery, combat and companion duties take precedence. Replies are unaffected.
//
// World thread only, all of it: this reads Players and pauses PlayerbotAI.
// --------------------------------------------------------------------------

// A reply to this person's say or yell is on its way from this bot: start the
// engagement, or refresh it. Replaces any other bot this person was engaged
// with, and any other person this bot was engaged with.
void Conversation_Engage(Player* bot, Player* person);

// Extend only the current engagement when its answer lands. Never reclaim a
// partner who has switched bots, or restart an engagement that has ended.
void Conversation_Refresh(Player* bot, Player* person);

// Is this bot engaged with this person right now?
bool Conversation_IsEngaged(Player* bot, Player* person);

// The bot this person is engaged with, if it is among the candidates.
Player* Conversation_PartnerAmong(Player* person, const std::vector<Player*>& candidates);

// Hold engaged bots still and release the ones whose conversation is over.
void Conversation_Update(uint32_t diff);

// Suppress unrelated ambient chatter while a bot is listening to a person.
bool Conversation_HasPartner(Player* bot);

// Number of engaged bots for the GM status command.
uint32_t Conversation_Count();

#endif // MOD_OLLAMA_CHAT_CONVERSATION_H
