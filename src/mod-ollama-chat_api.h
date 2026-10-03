#ifndef MOD_OLLAMA_CHAT_API_H
#define MOD_OLLAMA_CHAT_API_H

#include "mod-ollama-chat_capability.h"
#include <string>
#include <cstdint>

// Result of one generation call. The old API returned a bare string, so
// "server unreachable", "model refused to think" and "model said nothing"
// were indistinguishable -- all three surfaced as an empty reply and a
// generic log line.
struct OllamaApiResult
{
    std::string text;             // model output, reasoning already removed
    std::string thinking;         // native reasoning, kept only for debug logs
    bool        ok        = false;
    int         status    = 0;
    std::string error;
    uint64_t    latencyMs = 0;
    bool        thinkUsed = false;

    bool empty() const { return text.empty(); }
};

// --------------------------------------------------------------------------
// Immutable snapshot of everything a generation call needs from config.
//
// Worker threads used to read g_OllamaUrl / g_OllamaModel / g_OllamaSystemPrompt
// / g_OllamaStop / g_OllamaSeed directly. `.ollama reload` reassigns those
// std::strings on the world thread, which frees the old buffer underneath any
// worker mid-copy -- an access violation deep inside the HTTP client, with a
// stack that points at httplib rather than at the real cause.
//
// Config is now published once under a mutex and workers take a copy.
// --------------------------------------------------------------------------
struct OllamaEndpointSettings
{
    std::string url;
    std::string model;
    std::string systemPrompt;
    std::string memorySystemPrompt;
    std::string stop;
    std::string seed;

    uint32_t numPredict = 0;
    uint32_t numCtx     = 0;
    uint32_t numThreads = 0;

    float temperature   = 0.8f;
    float topP          = 0.95f;
    float repeatPenalty = 1.1f;

    // Optional sampling controls. Negative means "unset" -- the field is then
    // omitted from the request entirely and the model's own default applies.
    int32_t topK             = -1;
    float   minP             = -1.0f;
    float   presencePenalty  = -1000.0f;
    float   frequencyPenalty = -1000.0f;

    // The cheap lane. A classifier is not a voice: it wants a small fast model
    // answering in JSON, not the model that gives bots their character. When a
    // model is set here, request kinds that opt in are sent to it instead --
    // the url too, if one is given, so the lane can live on another backend.
    // Empty model = no lane, and every kind keeps using the fields above.
    std::string utilityUrl;
    std::string utilityModel;

    // The person lane: replies to what a real player said. Same shape as the
    // cheap lane, plus its own token cap, since a stronger model given room to
    // answer properly needs more than an ambient remark does.
    std::string replyUrl;
    std::string replyModel;
    uint32_t    replyNumPredict = 0;

    // Worker-side typing delay is read from the same immutable config snapshot.
    bool typingSimulation = false;
    bool deliverySplit = false;
    uint32_t deliveryMaxMessageBytes = 255;
    uint32_t typingBaseDelay = 1000;
    uint32_t typingDelayPerChar = 250;
    uint32_t typingMaxDelay = 0;
};

// Which model a request is sent to. The voice is the default; the person lane
// only applies when a reply model is configured.
enum class OllamaLane : uint8_t { Voice, Person };

// Republish from the g_Ollama* globals. Call on the world thread after config
// load or reload.
void OllamaConfig_Publish();

// Thread-safe copy for a worker.
OllamaEndpointSettings OllamaConfig_Snapshot();

// Perform one generation. Decides whether to think based on the configured
// policy and the probed model capability, and transparently retries once
// without thinking if Ollama rejects the request for asking.
//
// Blocking. Call from a worker thread, never from the world thread.
//
// voiceGuid identifies the character speaking, and is used only to give it its
// own small, fixed offset from the configured sampling distribution (plan 49
// item 3c). 0 means "not a character speaking" -- machinery calls leave it out
// and are never jittered.
OllamaApiResult QueryOllama(const std::string& prompt, OllamaRequestKind kind, uint64_t voiceGuid = 0,
                            OllamaLane lane = OllamaLane::Voice);

// Legacy shim: returns the text, or empty on any failure.
std::string QueryOllamaAPI(const std::string& prompt);

bool IsValidAPIResponse(const std::string& response);

#endif // MOD_OLLAMA_CHAT_API_H
