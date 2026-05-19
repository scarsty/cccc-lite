#pragma once
#include "DataPreparer.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace cccc
{

// 文本数据准备器（token 版本）
//
// 功能：
//   - 从 UTF-8 文本文件读取内容，按 Unicode code point 构建字符级词表
//   - 词表保存/加载到 <file>.vocab（二进制，每项 4 字节 uint32_t code point）
//   - 支持两种 X/Y 格式（由 dim0_[0] 是否等于 vocab_size_ 自动判断）：
//       raw_id  模式：X[t] = token_id（float），Y[t] = 下一个 token_id
//                     适合网络内部有 embedding 查表层的场景
//       one_hot 模式：X/Y 均为 (vocab_size × seq_len) 的展开 one-hot 向量
//                     适合直接接线性层/Transformer 的场景
//   - fillData0 每次随机采样 batch 条连续序列（长度 seq_len_），起点随机
//
// INI 配置示例：
//   [data_preparer]
//   mode    = txt
//   file    = corpus.txt
//   # raw_id 模式：dim 设为 (seq_len, 1, 1, batch)
//   # one_hot 模式：dim 设为 (vocab_size, seq_len, 1, batch)，vocab_size 需与实际词表匹配
class DataPreparerTxt : public DataPreparer
{
private:
    std::vector<int> tokens_;                         // 全文编码后的 token id 序列
    std::unordered_map<uint32_t, int> char_to_id_;    // Unicode code point → token id
    std::vector<uint32_t> id_to_char_;                // token id → Unicode code point

    int vocab_size_ = 0;
    int seq_len_ = 0;
    bool onehot_mode_ = false;    // true: one_hot 模式；false: raw_id 模式

public:
    DataPreparerTxt();
    virtual ~DataPreparerTxt();

    void init2() override;
    void fillData0() override;

    int getVocabSize() const { return vocab_size_; }

private:
    static std::vector<uint32_t> decodeUtf8(const std::string& s);
    void buildVocab(const std::vector<uint32_t>& codepoints, const std::string& vocab_file);
    void loadVocab(const std::string& vocab_file);
    void saveVocab(const std::string& vocab_file) const;

private:
    Random<double> rand_;
};

}    // namespace cccc