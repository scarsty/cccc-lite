# cccc 与 llama.cpp 对比

本文从设计理念、算子能力和 LLM 推理功能三个维度对比 cccc 与 llama.cpp。

---

## 1. 设计理念对比

| 维度 | cccc | llama.cpp |
|------|------|-----------|
| 定位 | 通用深度学习框架（支持训练 + 推理），网络结构由 cifa 脚本描述 | 专用 LLM 推理引擎，网络结构硬编码于 C++ |
| 网络描述方式 | cifa 脚本（运行时解析），模型结构对用户可见、可修改 | 硬编码 `llm_load_tensors` + `llm_build_*` C++ 函数 |
| 训练支持 | 支持反向传播、多种 solver（SGD / Adam 等） | 不支持训练 |
| 量化 | 无内置量化，当前仅支持 fp32 / fp16 / bf16 | 核心优势：2/3/4/5/6/8 bit GGUF 量化，多种 quant 格式 |
| 后端 | CUDA（CUBLAS / cuDNN）、HIP/ROCm（rocBLAS / MIOpen）、CPU | CUDA、Metal、Vulkan、CPU（BLAS 可选），支持异构 |
| 多 GPU | 数据并行（按 GPU 数切分 batch） | tensor 并行（权重分片，多 GPU 推理） |
| 内存管理 | `Matrix` 浅拷贝共享；KV cache 为命名矩阵，prefill/decode net 共享 | 统一 `ggml_context` 内存池，图节点引用 |
| 模型格式 | cccc 自定义 bin（named weight dump）；从 HuggingFace safetensors/bin 手动转换 | GGUF（自描述，含量化、超参、词表） |
| API | C++ 导出 DLL（`llm_chat` / `llm_chat_stream`），Python via SWIG | C 语言 `llama.h` API；Python / Go / Node 等多语言绑定 |

---

## 2. 算子对照

### 2.1 基础矩阵算子

| cccc cifa 函数 | llama.cpp 等价 | 说明 |
|---------------|---------------|------|
| `batchedMul(A, B, ta, tb)` | `ggml_mul_mat` | 批量矩阵乘，支持转置 |
| `elementMul(A, B)` | `ggml_mul` | 逐元素乘 |
| `add(A, B)` / `A + B` | `ggml_add` | 逐元素加 |
| `scale(X, s)` | `ggml_scale` | 标量乘 |
| `reshape(X, dims)` | `ggml_reshape_4d` | 形状变换（不复制数据） |
| `reshapeBatch(X, dims)` | `ggml_reshape_4d` | batch 维度参与的 reshape |
| `permute(X, perm)` | `ggml_permute` | 轴置换 |
| `concat(A, B, axis)` | `ggml_concat` | 指定轴拼接 |
| `tile(X, repeats)` | `ggml_repeat` | 沿各维重复 |
| `sliceW(X, start, len)` | `ggml_view_*` | 沿 width 轴切片 |
| `chunk(X, n, axis)` | `ggml_view_*` | 沿轴分块 |

### 2.2 归一化

| cccc | llama.cpp | 说明 |
|------|-----------|------|
| `rmsNorm(X, W)` | `ggml_rms_norm` + `ggml_mul` | RMS Norm，Llama 风格 |
| `layerNorm(X, W, b)` | `ggml_norm` + `ggml_mul` + `ggml_add` | Layer Norm，含 γ/β |
| `groupNorm(X, W, b, G)` | — | Group Norm，llama.cpp 无直接实现 |

### 2.3 激活函数

| cccc | llama.cpp | 说明 |
|------|-----------|------|
| `relu(X)` / `silu(X)` / `gelu(X)` / `tanh(X)` | `ggml_relu` / `ggml_silu` / `ggml_gelu` / `ggml_tanh` | 逐元素激活 |
| `active(X, type)` 统一入口 | 各独立 `ggml_*` 函数 | — |
| `softmaxChannel(X)` | `ggml_soft_max` | 沿 channel 维度 softmax |

### 2.4 注意力相关

| cccc | llama.cpp | 说明 |
|------|-----------|------|
| `attention(Q, K, V, dk, causal [,mask])` | `ggml_flash_attn_ext` / 手动 SDPA | cccc 为自实现 SDPA；llama.cpp 可选 Flash Attention |
| `rope(X, cos_tab, sin_tab)` | `ggml_rope_ext` | RoPE（half-rotate 风格，Llama 标准） |
| `rope2(X, cos_tab, sin_tab)` | `ggml_rope_ext` (interleaved) | RoPE interleaved 风格（GPT-NeoX） |
| `ropeCosTbl(T, d, base [,offset])` | 内部预计算 | cccc 显式生成 cos 表（可指定起始位置） |
| `ropeSinTbl(T, d, base [,offset])` | 内部预计算 | 同上，sin 表 |
| `kvcache(X, cache)` | `ggml_build_forward` KV 写入 | 写入共享 KV 缓存，自动按位置推进 |

### 2.5 Embedding 与输出

| cccc | llama.cpp | 说明 |
|------|-----------|------|
| `embed(ids, W)` | `ggml_get_rows` | token id → dense vector 查表 |
| `lmCrossEntropy(logits, targets)` | `ggml_cross_entropy_loss` | LM 训练 loss（含 softmax） |
| `crossEntropy(A, B)` | — | 一般 CE loss |

### 2.6 卷积 / 池化（CV 方向）

| cccc | llama.cpp | 说明 |
|------|-----------|------|
| `conv(X, W, [stride, pad])` | `ggml_conv_2d` | 2D 卷积 |
| `deconv(X, W)` | — | 转置卷积，llama.cpp 无 |
| `maxPool` / `averagePool` | `ggml_pool_2d` | 池化 |
| `pixelShuffle(X, r)` | — | 像素重组（超分），llama.cpp 无 |
| `upsample(X, factor)` | — | 上采样，llama.cpp 无 |

### 2.7 cccc 特有算子（llama.cpp 无直接对应）

| cccc | 用途 |
|------|------|
| `sinTimeEmbed(T, D)` | Transformer 正弦位置编码表生成 |
| `prependToken(X, tok)` | 在序列首部插入 token |
| `firstToken(X)` | 取序列第一个 token 输出 |
| `reparam(mu, logvar)` | VAE 重参数化采样 |
| `pixelShuffle` / `upsample` | 图像超分 / 生成任务 |

---

## 3. LLM 推理功能对比

| 功能 | cccc (`cccc-llm`) | llama.cpp |
|------|-------------------|-----------|
| KV cache 写入 | ✅ 命名 `Kcache_i`/`Vcache_i`，prefill/decode net 共享显存 | ✅ 连续 KV cache，支持 paged/ring 布局 |
| KV cache 重建（滑动窗口） | ✅ `rebuildKVCache`：超出 T_kv 时截断并重跑 prefill | ✅ 滑动窗口 / `n_past` 回滚 |
| Prefill + Decode 双网络 | ✅ `structure0`（多 token prefill）+ `structure1`（单 token decode） | ✅ 同一图，`n_tokens > 1` 为 prefill，`n_tokens=1` 为 decode |
| GQA（分组查询注意力） | ✅ cifa 脚本手动 tile 扩展 KV heads | ✅ 内置 GQA，`n_kv_heads` 配置 |
| RoPE 位置偏移（decode） | ✅ `setRopeOffset(pos)`，运行时动态设置 `window_[0]` | ✅ `n_past` 自动传入 RoPE kernel |
| Causal mask 位置偏移 | ✅ `setAttentionOffset(pos)` | ✅ 按 `n_past` 偏移 |
| 流式输出 | ✅ `llm_chat_stream` 回调 | ✅ `llama_decode` 逐步调用 |
| 贪心解码 | ✅ argmax | ✅ |
| Top-k / Top-p 采样 | ❌ 未实现 | ✅ `llama_sampler_chain` |
| 温度采样 | ❌ 未实现 | ✅ |
| Repetition penalty | ❌ 未实现 | ✅ |
| Beam search | ❌ 未实现 | ❌（官方也未提供） |
| 量化推理（GGUF） | ❌ 仅 fp32/fp16/bf16 | ✅ 2/3/4/5/6/8 bit |
| Flash Attention | ❌ 自实现 SDPA | ✅ 可选 Flash Attention 2 |
| LoRA 适配器加载 | ❌ 未实现 | ✅ |
| 多模态（Vision） | 可通过 cifa 扩展 | ✅ `llava` / `clip` 插件 |
| 工具调用 / Function Call | ✅ cccc-llm 层实现解析 | 需外部框架（llama.cpp 不内置） |
| Python 绑定 | ✅ SWIG 生成 | ✅ `llama-cpp-python` |
| 模型格式 | cccc 自定义 `.bin`（named weights） | GGUF（自描述，含词表） |
| 词表 / 分词器 | ✅ HuggingFace `tokenizer.json`（BPE）；字符级词表 | ✅ GGUF 内嵌 BPE / SentencePiece |

---

## 4. cccc 已知缺口与计划

| 缺口 | 影响 | 优先级 |
|------|------|--------|
| Top-k / Top-p / 温度采样 | 推理输出多样性差（目前仅 greedy） | 高 |
| Repetition penalty | 重复输出问题 | 高 |
| 量化（INT4/INT8） | 大模型显存占用高，无法在消费级 GPU 上运行大参数模型 | 高 |
| RoPE 动态 scaling（YaRN / Dynamic NTK） | 超过训练上下文长度时质量下降 | 中 |
| Flash Attention | 长序列显存和速度 | 中 |
| LoRA 加载 | 微调模型部署 | 中 |
| PAD token / label mask（训练） | padding 位置计入 loss，影响训练稳定性 | 低 |
| BOS token 注入（训练） | 部分模型要求显式 BOS | 低 |
