#pragma once
#include <string>

#ifdef _WIN32
#  ifdef CCCC_LLM_BUILDING
#    define CCCC_LLM_API __declspec(dllexport)
#  else
#    define CCCC_LLM_API __declspec(dllimport)
#  endif
#else
#  define CCCC_LLM_API
#endif

typedef void* LlmHandle;

// Callback invoked for each decoded token during streaming generation.
// token:    UTF-8 text of the current token (may be multi-byte).
// userdata: opaque pointer passed to llm_chat_stream.
typedef void (*LlmStreamCallback)(const std::string& token, void* userdata);

// Load model. ini_path: absolute or relative path to the prefill model .ini file.
// decode_ini_path: optional path to the decode (T=1, KV-cache) model .ini file.
// system_prompt: system prompt; defaults to "You are a helpful assistant.".
// Returns a handle on success, nullptr on failure.
CCCC_LLM_API LlmHandle llm_init(const std::string& ini_path, const std::string& system_prompt = "You are a helpful assistant.");

// Free all resources associated with a handle.
CCCC_LLM_API void llm_destroy(LlmHandle handle);

// Clear conversation history (system prompt is preserved).
CCCC_LLM_API void llm_reset(LlmHandle handle);

// Generate a reply to user_input and return the full reply as std::string.
// max_new_tokens <= 0 uses default (200).
// show_thinking: if non-zero, include <think>...</think> content in the reply.
CCCC_LLM_API std::string llm_chat(LlmHandle handle, const std::string& user_input, int max_new_tokens, int show_thinking);

// Streaming variant: callback is called once per decoded token.
// show_thinking: if zero, <think>...</think> tokens are suppressed before calling callback.
// Returns after generation is complete.
CCCC_LLM_API void llm_chat_stream(LlmHandle handle, const std::string& user_input, int max_new_tokens, int show_thinking, LlmStreamCallback callback, void* userdata);
