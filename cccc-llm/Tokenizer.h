#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

// Byte-level BPE tokenizer compatible with Qwen3 / HuggingFace tokenizer.json
class Tokenizer
{
public:
    bool load(const std::string& path);
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;

    int eos_token_id() const { return eos_id_; }
    int vocab_size() const { return (int)id_to_token_.size(); }

private:
    // GPT-2 byte <-> Unicode codepoint mapping (256 entries each way)
    uint32_t byte_to_cp_[256]{};
    std::unordered_map<uint32_t, uint8_t> cp_to_byte_;

    // BPE vocabulary
    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string> id_to_token_;

    // BPE merge ranks: key = token_a + '\0' + token_b, value = rank (lower = applied first)
    std::unordered_map<std::string, int> bpe_ranks_;

    // Special / added tokens
    std::unordered_map<std::string, int> added_tokens_;
    std::vector<std::string> added_tokens_sorted_; // sorted longest-first for greedy match

    int eos_id_ = 151645; // Qwen3 <|im_end|>

    void init_byte_tables();
    std::vector<std::string> bpe(const std::vector<std::string>& chars) const;
    std::vector<int> tokenize_segment(const std::string& text) const;
};
