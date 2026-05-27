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

// Append a tool-response turn to ctx (call after detecting <tool_call> in output).
// content: raw text to embed inside <tool_response>...</tool_response>.
CCCC_LLM_API void llm_inject_tool_response(LlmHandle handle, const std::string& content);

// Continue generation as the assistant without adding a new user turn.
// Use after injecting tool responses to get the model's follow-up.
CCCC_LLM_API void llm_continue_stream(LlmHandle handle, int max_new_tokens, int show_thinking, LlmStreamCallback callback, void* userdata);

// Disable thinking for all subsequent turns: inserts an empty <think></think> into
// the assistant header so the model skips internal reasoning.  Call once after
// llm_init() to apply to every chat/continue call (useful for agent tasks where
// thinking tokens waste KV-cache budget).
CCCC_LLM_API void llm_set_no_think(LlmHandle handle);

// After injecting a tool response, call this so KV-rebuild anchors to the full
// conversation (including the tool result) instead of just the original user turn.
CCCC_LLM_API void llm_update_anchor(LlmHandle handle);

// ─────────────────────────────────────────────────────────────────────────────
//  Stable Diffusion (z-image-turbo) API
// ─────────────────────────────────────────────────────────────────────────────

typedef void* SdHandle;

// Called after each DDIM sampling step.
// step: current step (1-based), total: total steps, userdata: opaque pointer.
typedef void (*SdProgressCallback)(int step, int total, void* userdata);

// Initialize Stable Diffusion pipeline from model directory.
// Returns handle on success, nullptr on failure.
CCCC_LLM_API SdHandle sd_init(const std::string& model_dir);

// Free all resources.
CCCC_LLM_API void sd_destroy(SdHandle handle);

// Text-to-image (DDIM sampling).
// guidance_scale is accepted but currently unused (CFG-free model).
// Returns 0 on success, non-zero on failure.
CCCC_LLM_API int sd_generate(SdHandle handle,
    const std::string& prompt,
    const std::string& output_path,
    int steps,
    float guidance_scale,
    int width,
    int height,
    int seed,
    SdProgressCallback callback,
    void* userdata);

// ─────────────────────────────────────────────────────────────────────────────
//  Coding Agent API
// ─────────────────────────────────────────────────────────────────────────────

typedef void* AgentHandle;

// Called for each decoded token during generation.
typedef void (*AgentStreamCallback)(const std::string& token, void* userdata);

// Called for tool names not built in (read_file / write_file / list_dir / run_cmd).
// Return non-empty string = tool result.  Return "" = unrecognized (error reported).
typedef std::string (*AgentCustomToolCallback)(const std::string& name,
    const std::string& args_json,
    void* userdata);

// System prompt to pass to llm_init so the model knows about available tools.
CCCC_LLM_API const char* agent_system_prompt();

// Create an agent wrapping an existing LlmHandle.
// The LlmHandle must remain valid until agent_destroy().
CCCC_LLM_API AgentHandle agent_create(LlmHandle llm);

// Free agent resources.  Does NOT destroy the underlying LlmHandle.
CCCC_LLM_API void agent_destroy(AgentHandle handle);

// Register a stream callback (called for every decoded token).
CCCC_LLM_API void agent_set_stream_callback(AgentHandle handle,
    AgentStreamCallback cb, void* userdata);

// Register a custom tool callback for tool names not built in.
CCCC_LLM_API void agent_set_custom_tool(AgentHandle handle,
    AgentCustomToolCallback cb, void* userdata);

// Run the agent on task.  Blocks until done or max_rounds exceeded.
// max_rounds <= 0 → default 16.   max_tokens <= 0 → default 4096.
CCCC_LLM_API void agent_run(AgentHandle handle,
    const std::string& task, int max_rounds, int max_tokens);
