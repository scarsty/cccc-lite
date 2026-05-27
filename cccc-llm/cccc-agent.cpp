#include "cccc-llm.h"
#include "Log.h"
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <vector>

#define NOMINMAX
#include <windows.h>

// Convert a system ANSI (CP_ACP / GBK on zh-CN Windows) string to UTF-8.
static std::string to_utf8(const std::string& s)
{
    if (s.empty()) return s;
    int wlen = MultiByteToWideChar(CP_ACP, 0, s.data(), (int)(s.size()), nullptr, 0);
    if (wlen <= 0) return s;
    std::wstring ws(wlen, L'\0');
    MultiByteToWideChar(CP_ACP, 0, s.data(), (int)(s.size()), ws.data(), wlen);
    int ulen = WideCharToMultiByte(CP_UTF8, 0, ws.data(), wlen, nullptr, 0, nullptr, nullptr);
    if (ulen <= 0) return s;
    std::string u(ulen, '\0');
    WideCharToMultiByte(CP_UTF8, 0, ws.data(), wlen, u.data(), ulen, nullptr, nullptr);
    return u;
}

// ─────────────────────────────────────────────────────────────────────────────
//  JSON helpers
// ─────────────────────────────────────────────────────────────────────────────

// Extract a JSON string value by key, handling \n \t \\ \" escapes.
static std::string json_get_string(const std::string& json, const std::string& key)
{
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) { return {}; }
    pos = json.find(':', pos);
    if (pos == std::string::npos) { return {}; }
    pos++;
    while (pos < json.size() &&
           (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' || json[pos] == '\t'))
    {
        pos++;
    }
    if (pos >= json.size() || json[pos] != '"') { return {}; }
    pos++;    // skip opening "
    std::string result;
    while (pos < json.size())
    {
        char c = json[pos++];
        if (c == '"') { break; }
        if (c == '\\' && pos < json.size())
        {
            char esc = json[pos++];
            switch (esc)
            {
            case 'n':  result += '\n'; break;
            case 't':  result += '\t'; break;
            case 'r':  result += '\r'; break;
            case '"':  result += '"';  break;
            case '\\': result += '\\'; break;
            case '/':  result += '/';  break;
            default:   result += esc;  break;
            }
        }
        else
        {
            result += c;
        }
    }
    return result;
}

// Extract the first balanced JSON object starting at or after start_pos.
// Correctly skips { } inside string literals.
static std::string extract_json_object(const std::string& text, size_t start_pos)
{
    size_t pos = text.find('{', start_pos);
    if (pos == std::string::npos) { return {}; }
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (size_t i = pos; i < text.size(); i++)
    {
        char c = text[i];
        if (escaped) { escaped = false; continue; }
        if (c == '\\' && in_string) { escaped = true; continue; }
        if (c == '"') { in_string = !in_string; continue; }
        if (in_string) { continue; }
        if (c == '{') { depth++; }
        else if (c == '}')
        {
            depth--;
            if (depth == 0) { return text.substr(pos, i - pos + 1); }
        }
    }
    return {};
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tool-call parsing
// ─────────────────────────────────────────────────────────────────────────────

struct ToolCall
{
    std::string name;
    std::string arguments;    // raw JSON object for "arguments"
};

static std::vector<ToolCall> parse_tool_calls(const std::string& text)
{
    static const std::string OPEN  = "<tool_call>";
    static const std::string CLOSE = "</tool_call>";

    std::vector<ToolCall> calls;
    size_t pos = 0;
    while (true)
    {
        auto start = text.find(OPEN, pos);
        if (start == std::string::npos) { break; }
        start += OPEN.size();
        while (start < text.size() &&
               (text[start] == '\n' || text[start] == '\r' || text[start] == ' '))
        {
            start++;
        }
        auto end = text.find(CLOSE, start);
        if (end == std::string::npos) { break; }
        size_t json_end = end;
        while (json_end > start &&
               (text[json_end - 1] == '\n' || text[json_end - 1] == '\r' || text[json_end - 1] == ' '))
        {
            json_end--;
        }
        std::string json_str = text.substr(start, json_end - start);

        ToolCall tc;
        tc.name = json_get_string(json_str, "name");
        auto arg_pos = json_str.find("\"arguments\"");
        if (arg_pos != std::string::npos)
        {
            tc.arguments = extract_json_object(json_str, arg_pos);
        }
        if (!tc.name.empty()) { calls.push_back(std::move(tc)); }
        pos = end + CLOSE.size();
    }
    return calls;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Built-in tools
// ─────────────────────────────────────────────────────────────────────────────

static std::string tool_read_file(const std::string& args)
{
    std::string path = json_get_string(args, "path");
    if (path.empty()) { return "[error: missing path argument]"; }
    std::ifstream f(path, std::ios::binary);
    if (!f) { return "[error: cannot open file: " + path + "]"; }
    std::string content((std::istreambuf_iterator<char>(f)), {});
    if (content.size() > 65536)
    {
        content = content.substr(0, 65536) + "\n...[file truncated at 64 KB]...";
    }
    return content;
}

static std::string tool_write_file(const std::string& args)
{
    std::string path    = json_get_string(args, "path");
    std::string content = json_get_string(args, "content");
    if (path.empty()) { return "[error: missing path argument]"; }
    std::filesystem::path p(path);
    if (p.has_parent_path())
    {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) { return "[error: cannot write file: " + path + "]"; }
    f.write(content.data(), (std::streamsize)content.size());
    return std::format("[ok: wrote {} bytes to {}]", content.size(), path);
}

static std::string tool_list_dir(const std::string& args)
{
    std::string path = json_get_string(args, "path");
    if (path.empty()) { path = "."; }
    std::error_code ec;
    std::string result;
    for (auto& entry : std::filesystem::directory_iterator(path, ec))
    {
        result += entry.path().filename().string();
        if (entry.is_directory()) { result += '/'; }
        result += '\n';
    }
    if (ec) { return "[error: " + ec.message() + "]"; }
    return result.empty() ? "[empty directory]" : result;
}

static std::string tool_run_cmd(const std::string& args)
{
    std::string cmd = json_get_string(args, "cmd");
    if (cmd.empty()) { return "[error: missing cmd argument]"; }
    std::string full_cmd = "cmd /c " + cmd + " 2>&1";
    FILE* p = _popen(full_cmd.c_str(), "r");
    if (!p) { return "[error: failed to start process]"; }
    std::string out;
    char buf[4096];
    while (fgets(buf, sizeof(buf), p))
    {
        out += buf;
        if (out.size() > 65536)
        {
            out += "\n...[output truncated]...";
            break;
        }
    }
    int rc = _pclose(p);
    out += std::format("\n[exit code: {}]", rc);
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  AgentSession
// ─────────────────────────────────────────────────────────────────────────────

struct AgentSession
{
    LlmHandle llm = nullptr;
    AgentStreamCallback     stream_cb   = nullptr;
    void*                   stream_ud   = nullptr;
    AgentCustomToolCallback custom_tool = nullptr;
    void*                   custom_ud   = nullptr;
};

// ─────────────────────────────────────────────────────────────────────────────
//  System prompt (Qwen3 tool-use format)
// ─────────────────────────────────────────────────────────────────────────────

static const char* SYSTEM_PROMPT = R"(You are a helpful agent. Use the available tools to answer the user's question. Only call tools when necessary to answer the question.

<tools>
{"type":"function","function":{"name":"read_file","description":"Read the full content of a file. Use this to inspect file contents.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}
{"type":"function","function":{"name":"write_file","description":"Write file.","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}
{"type":"function","function":{"name":"list_dir","description":"List directory contents.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}
{"type":"function","function":{"name":"run_cmd","description":"Run a build or system command. Do NOT use this to read files; use read_file instead.","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}}
</tools>

Use <tool_call>{"name":"...","arguments":{...}}</tool_call> to call a tool.

On Windows, C/C++ compilers are not in PATH by default. To compile C/C++ code, use a helper batch file:
1. Use write_file to create a compile.bat (use the full absolute path, e.g. C:\work\compile.bat):
   @echo off\r\ncall "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64\r\ncl /EHsc /Fe:program.exe source.cpp\r\nprogram.exe
2. Use run_cmd to execute it by full absolute path: C:\work\compile.bat
Note: VS2026 vcvarsall path is "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat". Use "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath to find the correct VS install path.)";

// ─────────────────────────────────────────────────────────────────────────────
//  Exported API
// ─────────────────────────────────────────────────────────────────────────────

CCCC_LLM_API const char* agent_system_prompt()
{
    return SYSTEM_PROMPT;
}

CCCC_LLM_API AgentHandle agent_create(LlmHandle llm)
{
    auto* s  = new AgentSession();
    s->llm   = llm;
    return s;
}

CCCC_LLM_API void agent_destroy(AgentHandle handle)
{
    delete (AgentSession*)(handle);
}

CCCC_LLM_API void agent_set_stream_callback(AgentHandle handle,
    AgentStreamCallback cb, void* userdata)
{
    auto* s = (AgentSession*)(handle);
    if (!s) { return; }
    s->stream_cb = cb;
    s->stream_ud = userdata;
}

CCCC_LLM_API void agent_set_custom_tool(AgentHandle handle,
    AgentCustomToolCallback cb, void* userdata)
{
    auto* s = (AgentSession*)(handle);
    if (!s) { return; }
    s->custom_tool = cb;
    s->custom_ud   = userdata;
}

CCCC_LLM_API void agent_run(AgentHandle handle,
    const std::string& task, int max_rounds, int max_tokens)
{
    auto* s = (AgentSession*)(handle);
    if (!s || !s->llm) { return; }
    if (max_rounds <= 0) { max_rounds = 16; }
    if (max_tokens <= 0) { max_tokens = 4096; }

    // Disable thinking to conserve KV-cache budget: with T_kv=1024 and a 321-token
    // system prompt, thinking chains would exhaust the context window before the
    // model can output a tool call or an answer.
    llm_set_no_think(s->llm);

    // Per-run log file written to the current working directory.
    std::ofstream logf("agent_log.txt", std::ios::binary | std::ios::trunc);
    if (logf)
    {
        logf.write("\xEF\xBB\xBF", 3); // UTF-8 BOM
        logf << "=== cccc-agent log ===\n"
             << "Task: " << to_utf8(task) << "\n\n"
             << "========================================\n"
             << "SYSTEM PROMPT (Qwen3 chat format)\n"
             << "========================================\n"
             << "<|im_start|>system\n"
             << SYSTEM_PROMPT
             << "<|im_end|>\n"
             << "========================================\n\n";
        logf.flush();
    }

    // Collect generated text, forward to stream callback, and mirror to log file.
    struct StreamCtx
    {
        std::string      text;
        AgentSession*    agent;
        std::ofstream*   logf;
    };
    auto stream_fn = [](const std::string& tok, void* ud)
    {
        auto* ctx = (StreamCtx*)(ud);
        ctx->text += tok;
        if (ctx->agent->stream_cb)
        {
            ctx->agent->stream_cb(tok, ctx->agent->stream_ud);
        }
        if (ctx->logf && ctx->logf->is_open())
        {
            ctx->logf->write(tok.data(), (std::streamsize)tok.size());
        }
    };

    // Tool results injected into the LLM are capped so they don't exceed the
    // KV-cache window; the full result is still written to the log file.
    static const size_t MAX_TOOL_RESULT_TOKENS = 800;     // ~200 tokens (safe margin for T_kv=1024)

    std::string prev_call_key;   // "name|args" of the last executed tool call

    for (int round = 0; round < max_rounds; round++)
    {
        if (logf) { logf << "\n--- Round " << round + 1 << " ---\n"; logf.flush(); }

        StreamCtx ctx{ {}, s, logf ? &logf : nullptr };
        if (round == 0)
        {
            // Prepend the current working directory so the model can use correct
            // relative paths without guessing the environment.
            // Convert from system ANSI (GBK on zh-CN) to UTF-8 so the tokenizer
            // receives valid Unicode text.
            std::string cwd_task = "[Working directory: "
                + to_utf8(std::filesystem::current_path().string())
                + "]\n\n" + to_utf8(task);
            if (logf)
            {
                logf << "► USER TURN (发给模型的用户消息):\n"
                     << "<|im_start|>user\n" << cwd_task << "<|im_end|>\n"
                     << "<|im_start|>assistant\n<think>\n</think>\n"
                     << "────────────────────────────────────────\n"
                     << "◀ ASSISTANT OUTPUT (模型生成):\n";
                logf.flush();
            }
            llm_chat_stream(s->llm, cwd_task, max_tokens, 0, stream_fn, &ctx);
        }
        else
        {
            if (logf)
            {
                logf << "► ASSISTANT CONTINUES (继续生成，无新用户消息):\n"
                     << "<|im_start|>assistant\n<think>\n</think>\n"
                     << "────────────────────────────────────────\n"
                     << "◀ ASSISTANT OUTPUT (模型生成):\n";
                logf.flush();
            }
            llm_continue_stream(s->llm, max_tokens, 0, stream_fn, &ctx);
        }
        if (logf) { logf << "\n────────────────────────────────────────\n"; logf.flush(); }

        auto calls = parse_tool_calls(ctx.text);
        if (calls.empty()) { break; }    // no tool calls → agent finished

        // Execute only the FIRST tool call and start a new round immediately.
        // Injecting multiple results in one round inflates ctx_ids and can push
        // the total well past T_kv, causing rebuild-induced context loss.
        {
            auto& call = calls[0];

            // Detect repeated identical tool calls (e.g. the same failing command).
            // Inject a guidance note instead of re-running to break the loop.
            std::string call_key = call.name + "|" + call.arguments;
            if (call_key == prev_call_key)
            {
                const std::string hint =
                    "[system: The same tool call was tried in the previous round and did not help. "
                    "Please try a different approach. "
                    "On Windows, use 'findstr' instead of 'grep', and 'type' instead of 'cat'. "
                    "Or use read_file to read file contents directly.]";
                if (logf)
                {
                    logf << "\n[Tool: " << call.name << " (skipped - repeated call)]\n"
                         << "[Result]\n" << hint << "\n";
                    logf.flush();
                }
                cccc::LOG("\n[Tool: {} SKIPPED - repeated call, injecting hint]\n", call.name);
                llm_inject_tool_response(s->llm, hint);
                continue;
            }
            prev_call_key = call_key;
            cccc::LOG("\n[Tool: {}]\n", call.name);
            if (logf)
            {
                logf << "\n[Tool: " << call.name << "]\n"
                     << "[Args] " << call.arguments << "\n";
                logf.flush();
            }

            std::string result;
            if      (call.name == "read_file")  { result = tool_read_file(call.arguments); }
            else if (call.name == "write_file") { result = tool_write_file(call.arguments); }
            else if (call.name == "list_dir")   { result = tool_list_dir(call.arguments); }
            else if (call.name == "run_cmd")    { result = tool_run_cmd(call.arguments); }
            else if (s->custom_tool)
            {
                result = s->custom_tool(call.name, call.arguments, s->custom_ud);
                if (result.empty())
                {
                    result = "[error: unknown tool '" + call.name + "']";
                }
            }
            else
            {
                result = "[error: unknown tool '" + call.name + "']";
            }

            // Write full result to log file.
            if (logf)
            {
                // Show the Qwen3 tool_response format that actually gets injected
                // into ctx_ids (same as what llm_inject_tool_response generates).
                std::string truncated = result;
                if (truncated.size() > MAX_TOOL_RESULT_TOKENS)
                    truncated = truncated.substr(0, MAX_TOOL_RESULT_TOKENS) + "\n...[truncated]...";

                logf << "\n► TOOL RESULT (注入模型的 token 序列):\n"
                     << "<|im_start|>tool\n<tool_response>\n"
                     << truncated
                     << "\n</tool_response>\n<|im_end|>\n"
                     << "────────────────────────────────────────\n"
                     << "[Full raw result]\n" << result << "\n";
                logf.flush();
            }

            // Show brief preview on the terminal.
            std::string preview = result.substr(0, 200);
            if (result.size() > 200) { preview += "..."; }
            cccc::LOG("[Result] {}\n\n", preview);

            // Truncate before injecting into LLM context to stay within T_kv.
            if (result.size() > MAX_TOOL_RESULT_TOKENS)
            {
                result = result.substr(0, MAX_TOOL_RESULT_TOKENS) + "\n...[truncated]...";
            }
            llm_inject_tool_response(s->llm, result);
        }
    }

    if (logf) { logf << "\n--- Done ---\n"; }
}
