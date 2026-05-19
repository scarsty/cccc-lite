#include "Tokenizer.h"
#include "FakeJson.h"
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstring>
#include <fstream>

// ─────────────────────────── UTF-8 helpers ───────────────────────────────────

static std::string cp_to_utf8(uint32_t cp)
{
    std::string s;
    if (cp < 0x80)
    {
        s += (char)cp;
    }
    else if (cp < 0x800)
    {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    }
    else if (cp < 0x10000)
    {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
    else
    {
        s += (char)(0xF0 | (cp >> 18));
        s += (char)(0x80 | ((cp >> 12) & 0x3F));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
    return s;
}

// Decode next UTF-8 codepoint from s at pos, advance pos
static uint32_t utf8_next(const std::string& s, size_t& pos)
{
    uint8_t c = (uint8_t)s[pos];
    uint32_t cp;
    int len;
    if (c < 0x80)
    {
        cp = c;
        len = 1;
    }
    else if (c < 0xE0)
    {
        cp = c & 0x1F;
        len = 2;
    }
    else if (c < 0xF0)
    {
        cp = c & 0x0F;
        len = 3;
    }
    else
    {
        cp = c & 0x07;
        len = 4;
    }
    for (int i = 1; i < len && pos + i < s.size(); i++)
    {
        cp = (cp << 6) | ((uint8_t)s[pos + i] & 0x3F);
    }
    pos += len;
    return cp;
}

// ─────────────────────── Byte <-> Unicode codepoint ──────────────────────────

void Tokenizer::init_byte_tables()
{
    // Mirrors Python's bytes_to_unicode() used by GPT-2 / tiktoken / Qwen3.
    // Printable bytes (33-126, 161-172, 174-255) map to the same codepoint.
    // The remaining 68 bytes map to U+0100..U+0143 in iteration order.
    bool in_bs[256] = {};
    std::vector<int> bs, cs;

    auto add_printable = [&](int lo, int hi)
    {
        for (int i = lo; i <= hi; i++)
        {
            in_bs[i] = true;
            bs.push_back(i);
            cs.push_back(i);
        }
    };
    add_printable(33, 126);
    add_printable(161, 172);
    add_printable(174, 255);

    int n = 0;
    for (int b = 0; b < 256; b++)
    {
        if (!in_bs[b])
        {
            bs.push_back(b);
            cs.push_back(256 + n++);
        }
    }

    for (int i = 0; i < 256; i++)
    {
        byte_to_cp_[(uint8_t)bs[i]] = (uint32_t)cs[i];
        cp_to_byte_[(uint32_t)cs[i]] = (uint8_t)bs[i];
    }
}

// ────────────────────────────── load() ───────────────────────────────────────

bool Tokenizer::load(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
    {
        return false;
    }
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    init_byte_tables();

    auto root = FakeJson::parse(content);
    if (!root.isMap())
    {
        return false;
    }

    // added_tokens: [{id: int, content: string, ...}, ...]
    if (root.exist("added_tokens") && root["added_tokens"].isVector())
    {
        for (auto& item : root["added_tokens"].get<std::vector<FakeJson>>())
        {
            if (!item.isMap() || !item.exist("id") || !item.exist("content"))
            {
                continue;
            }
            int id = item["id"].toInt();
            std::string tok = item["content"].toString();
            added_tokens_[tok] = id;
            if (tok == "<|endoftext|>")
            {
                eos_id_ = id;
            }
        }
    }

    // model.vocab: {"token": id, ...}  and  model.merges: [[a, b], ...]
    if (root.exist("model") && root["model"].isMap())
    {
        auto& model = root["model"];

        if (model.exist("vocab") && model["vocab"].isMap())
        {
            for (auto& [token, id_fj] : model["vocab"].get<std::map<std::string, FakeJson>>())
            {
                if (!id_fj.isInt())
                {
                    continue;
                }
                int id = id_fj.toInt();
                vocab_[token] = id;
                if (id >= (int)id_to_token_.size())
                {
                    id_to_token_.resize(id + 1);
                }
                id_to_token_[id] = token;
            }
        }

        if (model.exist("merges") && model["merges"].isVector())
        {
            int rank = 0;
            for (auto& pair_fj : model["merges"].get<std::vector<FakeJson>>())
            {
                if (!pair_fj.isVector())
                {
                    continue;
                }
                auto pair = pair_fj.get<std::vector<FakeJson>>();
                if ((int)pair.size() < 2 || !pair[0].isString() || !pair[1].isString())
                {
                    continue;
                }
                bpe_ranks_[pair[0].toString() + '\0' + pair[1].toString()] = rank++;
            }
        }
    }

    // Merge added tokens into id_to_token_ (they may extend beyond BPE vocab)
    for (const auto& kv : added_tokens_)
    {
        int id = kv.second;
        if (id >= (int)id_to_token_.size())
        {
            id_to_token_.resize(id + 1);
        }
        if (id_to_token_[id].empty())
        {
            id_to_token_[id] = kv.first;
        }
    }

    // Sort added_tokens by length descending for greedy matching in encode()
    for (const auto& kv : added_tokens_)
    {
        added_tokens_sorted_.push_back(kv.first);
    }
    std::sort(added_tokens_sorted_.begin(), added_tokens_sorted_.end(),
        [](const std::string& a, const std::string& b)
        {
            return a.size() > b.size();
        });

    return !vocab_.empty();
}

// ────────────────────────────── BPE algorithm ────────────────────────────────

std::vector<std::string> Tokenizer::bpe(const std::vector<std::string>& chars) const
{
    std::vector<std::string> tokens = chars;
    if (tokens.size() <= 1)
    {
        return tokens;
    }

    while (true)
    {
        // Find the pair with the lowest merge rank
        int best_rank = INT_MAX;
        int best_i = -1;
        for (int i = 0; i < (int)tokens.size() - 1; i++)
        {
            auto it = bpe_ranks_.find(tokens[i] + '\0' + tokens[i + 1]);
            if (it != bpe_ranks_.end() && it->second < best_rank)
            {
                best_rank = it->second;
                best_i = i;
            }
        }
        if (best_i == -1)
        {
            break;
        }

        // Merge ALL occurrences of this pair (left to right)
        const std::string a = tokens[best_i];
        const std::string b = tokens[best_i + 1];
        std::vector<std::string> next;
        next.reserve(tokens.size());
        for (int i = 0; i < (int)tokens.size();)
        {
            if (i + 1 < (int)tokens.size() && tokens[i] == a && tokens[i + 1] == b)
            {
                next.push_back(a + b);
                i += 2;
            }
            else
            {
                next.push_back(tokens[i]);
                i++;
            }
        }
        tokens = std::move(next);
    }
    return tokens;
}

// ─────────────────── tokenize_segment (one non-special chunk) ────────────────

std::vector<int> Tokenizer::tokenize_segment(const std::string& text) const
{
    if (text.empty())
    {
        return {};
    }

    // Step 1: byte-encode: each input byte → corresponding Unicode char (as UTF-8)
    std::string byte_encoded;
    byte_encoded.reserve(text.size() * 2);
    for (uint8_t b : text)
    {
        byte_encoded += cp_to_utf8(byte_to_cp_[b]);
    }

    // Step 2: split into individual Unicode chars (one per original byte)
    std::vector<std::string> chars;
    chars.reserve(text.size());
    {
        size_t pos = 0;
        while (pos < byte_encoded.size())
        {
            size_t start = pos;
            utf8_next(byte_encoded, pos);
            chars.push_back(byte_encoded.substr(start, pos - start));
        }
    }

    // Step 3: apply BPE merges
    auto tokens = bpe(chars);

    // Step 4: look up ids
    std::vector<int> ids;
    ids.reserve(tokens.size());
    for (const auto& tok : tokens)
    {
        auto it = vocab_.find(tok);
        if (it != vocab_.end())
        {
            ids.push_back(it->second);
        }
        // else: unknown token – skip (should not happen for valid UTF-8)
    }
    return ids;
}

// ──────────────────────────── encode() ───────────────────────────────────────

std::vector<int> Tokenizer::encode(const std::string& text) const
{
    std::vector<int> result;
    size_t pos = 0;
    const size_t len = text.size();

    while (pos < len)
    {
        // Try to match a special/added token at the current position
        bool matched = false;
        for (const auto& st : added_tokens_sorted_)
        {
            if (len - pos >= st.size()
                && text.compare(pos, st.size(), st) == 0)
            {
                result.push_back(added_tokens_.at(st));
                pos += st.size();
                matched = true;
                break;
            }
        }
        if (matched)
        {
            continue;
        }

        // Find the next special token to know where this segment ends
        size_t next_special = len;
        for (const auto& st : added_tokens_sorted_)
        {
            size_t found = text.find(st, pos);
            if (found != std::string::npos && found < next_special)
            {
                next_special = found;
            }
        }

        // BPE-encode the segment between pos and next_special
        auto seg_ids = tokenize_segment(text.substr(pos, next_special - pos));
        result.insert(result.end(), seg_ids.begin(), seg_ids.end());
        pos = next_special;
    }
    return result;
}

// ──────────────────────────── decode() ───────────────────────────────────────

std::string Tokenizer::decode(const std::vector<int>& ids) const
{
    std::string result;
    for (int id : ids)
    {
        if (id < 0 || id >= (int)id_to_token_.size())
        {
            continue;
        }
        const std::string& token = id_to_token_[id];

        // Map each Unicode char back to the original byte
        size_t pos = 0;
        bool all_mapped = true;
        std::string seg;
        while (pos < token.size())
        {
            uint32_t cp = utf8_next(token, pos);
            auto it = cp_to_byte_.find(cp);
            if (it != cp_to_byte_.end())
            {
                seg += (char)it->second;
            }
            else
            {
                // This codepoint is not in our byte alphabet
                // (e.g. a special token like <|im_end|>). Skip.
                all_mapped = false;
                break;
            }
        }
        if (all_mapped)
        {
            result += seg;
        }
        // Special tokens are silently skipped in output
    }
    return result;
}
