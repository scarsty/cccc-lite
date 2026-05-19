#include "DataPreparerTxt.h"
#include "Log.h"
#include "VectorMath.h"
#include "filefunc.h"
#include <algorithm>

namespace cccc
{

DataPreparerTxt::DataPreparerTxt()
{
}

DataPreparerTxt::~DataPreparerTxt()
{
}

// UTF-8 字节流 → Unicode code point 序列
// 跳过无效字节序列，保证不越界
std::vector<uint32_t> DataPreparerTxt::decodeUtf8(const std::string& s)
{
    std::vector<uint32_t> result;
    result.reserve(s.size());
    const uint8_t* p = reinterpret_cast<const uint8_t*>(s.data());
    const uint8_t* end = p + s.size();
    while (p < end)
    {
        uint32_t cp = 0;
        if (*p < 0x80)
        {
            cp = *p++;
        }
        else if ((*p & 0xE0) == 0xC0 && p + 1 < end && (*(p + 1) & 0xC0) == 0x80)
        {
            cp = ((*p & 0x1F) << 6) | (*(p + 1) & 0x3F);
            p += 2;
        }
        else if ((*p & 0xF0) == 0xE0 && p + 2 < end && (*(p + 1) & 0xC0) == 0x80 && (*(p + 2) & 0xC0) == 0x80)
        {
            cp = ((*p & 0x0F) << 12) | ((*(p + 1) & 0x3F) << 6) | (*(p + 2) & 0x3F);
            p += 3;
        }
        else if ((*p & 0xF8) == 0xF0 && p + 3 < end && (*(p + 1) & 0xC0) == 0x80 && (*(p + 2) & 0xC0) == 0x80 && (*(p + 3) & 0xC0) == 0x80)
        {
            cp = ((*p & 0x07) << 18) | ((*(p + 1) & 0x3F) << 12) | ((*(p + 2) & 0x3F) << 6) | (*(p + 3) & 0x3F);
            p += 4;
        }
        else
        {
            p++;    // 跳过无效字节
            continue;
        }
        result.push_back(cp);
    }
    return result;
}

void DataPreparerTxt::saveVocab(const std::string& vocab_file) const
{
    filefunc::writeVectorToFile(id_to_char_, vocab_file);
    LOG("Saved vocabulary ({} tokens) to \"{}\"\n", vocab_size_, vocab_file);
}

void DataPreparerTxt::loadVocab(const std::string& vocab_file)
{
    filefunc::readFileToVector(vocab_file, id_to_char_);
    vocab_size_ = (int)id_to_char_.size();
    char_to_id_.clear();
    for (int i = 0; i < vocab_size_; i++)
        char_to_id_[id_to_char_[i]] = i;
    LOG("Loaded vocabulary ({} tokens) from \"{}\"\n", vocab_size_, vocab_file);
}

// 从 code point 序列构建词表：排序去重后建立双向映射，并保存到文件
void DataPreparerTxt::buildVocab(const std::vector<uint32_t>& codepoints, const std::string& vocab_file)
{
    id_to_char_ = codepoints;
    std::sort(id_to_char_.begin(), id_to_char_.end());
    id_to_char_.erase(std::unique(id_to_char_.begin(), id_to_char_.end()), id_to_char_.end());
    vocab_size_ = (int)id_to_char_.size();
    char_to_id_.clear();
    for (int i = 0; i < vocab_size_; i++)
        char_to_id_[id_to_char_[i]] = i;
    LOG("Built vocabulary: {} unique tokens\n", vocab_size_);
    saveVocab(vocab_file);
}

void DataPreparerTxt::init2()
{
    std::string filename = option_->getString(section_, "file", "file.txt");
    std::string content = filefunc::readFileToString(filename);
    if (content.empty())
    {
        LOG("Warning: text file \"{}\" is empty or not found\n", filename);
        return;
    }

    // 解码 UTF-8 → code point 序列
    auto codepoints = decodeUtf8(content);
    LOG("Text: \"{}\" — {} code points\n", filename, codepoints.size());

    // 加载已有词表，或从文本中构建新词表
    std::string vocab_file = filefunc::getFileMainName(filename) + ".vocab";
    if (filefunc::fileExist(vocab_file))
    {
        loadVocab(vocab_file);
        // 对词表中没有的字符追加到末尾（增量扩展）
        bool extended = false;
        for (auto cp : codepoints)
        {
            if (char_to_id_.find(cp) == char_to_id_.end())
            {
                char_to_id_[cp] = vocab_size_++;
                id_to_char_.push_back(cp);
                extended = true;
            }
        }
        if (extended)
        {
            LOG("Vocabulary extended to {} tokens\n", vocab_size_);
            saveVocab(vocab_file);
        }
    }
    else
    {
        buildVocab(codepoints, vocab_file);
    }

    // 将全文编码为 token id 序列
    tokens_.clear();
    tokens_.reserve(codepoints.size());
    for (auto cp : codepoints)
        tokens_.push_back(char_to_id_[cp]);

    // 根据 dim0_[0] 与 vocab_size_ 是否相同推断输入格式
    // one_hot 模式：dim0_ = {vocab_size, seq_len, ...}，第 0 维必须恰好等于词表大小
    // raw_id  模式：dim0_ = {seq_len, ...}，X 每位存一个 token id
    int input_size = VectorMath::multiply(dim0_, (int)dim0_.size() - 1);
    if (!dim0_.empty() && dim0_[0] == vocab_size_)
    {
        onehot_mode_ = true;
        seq_len_ = (dim0_.size() >= 2) ? dim0_[1] : 1;
        LOG("Token mode: one_hot, seq_len={}, vocab_size={}\n", seq_len_, vocab_size_);
    }
    else
    {
        onehot_mode_ = false;
        seq_len_ = input_size;
        LOG("Token mode: raw_id,  seq_len={}, vocab_size={}\n", seq_len_, vocab_size_);
    }
}

void DataPreparerTxt::fillData0()
{
    rand_.set_seed();
    if (tokens_.empty() || seq_len_ <= 0)
        return;

    int total = (int)tokens_.size();
    int n_samples = X.getNumber();

    // 每条样本至少需要 seq_len+1 个 token（最后一个 token 作为 Y 的最终目标）
    if (total < seq_len_ + 1)
    {
        LOG("Warning: text too short ({} tokens) for seq_len={}\n", total, seq_len_);
        return;
    }
    int max_start = total - seq_len_ - 1;

    for (int idx = 0; idx < n_samples; idx++)
    {
        int start = (int)(rand_.rand() * max_start);

        if (onehot_mode_)
        {
            // one_hot 模式：X 和 Y 各是 (vocab_size × seq_len) 展开的 one-hot 矩阵
            // 内存布局：X.row() = vocab_size * seq_len，按 (token_dim, position) 列主序
            // X[v + t * vocab_size_, idx] = 1.0f 当 tokens_[start+t] == v
            // Y[v + t * vocab_size_, idx] = 1.0f 当 tokens_[start+t+1] == v
            for (int t = 0; t < seq_len_; t++)
            {
                int x_id = tokens_[start + t];
                int y_id = tokens_[start + t + 1];
                for (int v = 0; v < vocab_size_; v++)
                {
                    X.setData(v + t * vocab_size_, idx, (v == x_id) ? 1.0f : 0.0f);
                    Y.setData(v + t * vocab_size_, idx, (v == y_id) ? 1.0f : 0.0f);
                }
            }
        }
        else
        {
            // raw_id 模式：X[t] = token_id（float），Y[t] = 下一个 token_id
            for (int t = 0; t < seq_len_; t++)
            {
                X.setData(t, idx, (float)tokens_[start + t]);
                Y.setData(t, idx, (float)tokens_[start + t + 1]);
            }
        }
    }
}

}    // namespace cccc