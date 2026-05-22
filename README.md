# CCCC
<img src='https://raw.githubusercontent.com/scarsty/neural-demo/master/logo.png'>

cccc为libwill的核心部分，负责神经网络的训练与推理。

英文代号“will”意为“心愿”。

## 简介

cccc是同时支持第一代基于层和第二代基于操作的神经网络工具包，但是安装和使用远比同类工具简单。

cccc正式支持Windows，且以Windows为主要开发平台。可以在不需要nccl的情况下进行并行。

cccc的设计同时支持动态图和静态图。

cccc同时支持N卡和A卡，无需为不同的显卡平台编译两个版本。甚至可以在同一电脑上安装两种显卡并行计算（并不是建议你这样做）。

cccc的功能并不及其他的开源平台，其特色是简单的配置和单机下的速度，请酌情使用。

cccc同时支持大语言模型（LLM）的推理，内置字节级BPE分词器（兼容Qwen3等主流模型），支持多轮对话、思维链过滤和流式逐词输出。cccc-llm动态库还集成了扩散模型图像生成（Diffusion Model）和编码智能体（Coding Agent）功能。

## 大语言模型推理

cccc通过cccc-llm动态库（cccc-llm.dll）提供LLM推理接口，由cccc-windows调用，实现交互式对话。

目前已验证支持以下模型：

| 模型 | 参数量 | 特性 |
|------|--------|------|
| Qwen3 系列（如 Qwen3-0.5B） | 0.5B～8B | 通用对话，支持思维链（thinking） |
| Hy-MT2-7B（混元机器翻译） | 7B | 专用翻译模型，BF16，GQA，use_qk_norm，tie_word_embeddings |

需先将 HuggingFace 格式的权重转换为 cccc 格式，已转换好的权重在 <https://huggingface.co/scarsty/cccc-llm>。

### 运行示例

```shell
# Qwen3 交互式多轮对话
cccc-windows --llm -c work\qwen3\qwen3_0.5b_fp16.ini

# Hy-MT2-7B 翻译（BF16）
cccc-windows --llm -c work\Hy-MT2\hy_mt2_7b_fp16.ini
```

模型目录下应有 `tokenizer.json` 文件（路径可写相对路径，cccc 会自动从 ini 所在目录查找）。

#### Hy-MT2-7B 说明

Hy-MT2-7B 是腾讯混元专用机器翻译模型（HunYuanDenseV1 架构），支持多语言互译。对话模板：

```
用户：<|startoftext|>{输入}<|extra_0|>
回复：{翻译结果}<|eos|>
```

## 图像生成（Diffusion Model）

cccc通过cccc-llm动态库（cccc-llm.dll）提供扩散模型图像生成接口，支持文本到图像（Text-to-Image）的推理。

目前已验证支持 z-image-turbo 模型（基于 FLUX.1 的蒸馏版本）。完整模型和配置文件已发布在 <https://huggingface.co/scarsty/cccc-sd>。

模型由<https://github.com/nihui/zimage-ncnn-vulkan>提供，已转换为cccc格式并验证推理结果基本一致。

### 运行示例

#### 交互式生成

```shell
# 进入模型目录，交互式输入提示词
cd path/to/z-image-turbo
cccc-windows --sd -c .
# 输入提示词，按 Enter 生成图片，输入 /quit 退出
```

#### 命令行单次生成

```shell
cd path/to/z-image-turbo
# 不指定 seed：每次随机种子
echo "serene landscape with mountains" | cccc-windows --sd -c .

# 指定 seed：可复现结果
cccc-windows --sd -c . -p "serene landscape with mountains" -o seed_42.png --seed 42
```

种子参数说明：

- `--seed`（短参数 `-e`）为可选项。
- 不传 `--seed` 时，程序每次生成都会自动使用随机种子。
- 传入固定 seed 时，同提示词可复现一致结果。

### 配置文件说明

- `net_*.ini`：各子网络配置文件（文本编码器、去噪网络等）
- `vocab.txt`、`merges.txt`：BPE分词器文件
- `z_image_turbo_*.cccc.bin`：预训练模型权重文件

## 编码智能体（Coding Agent）

cccc-llm动态库（cccc-llm.dll）内置编码智能体，支持通过大语言模型自动完成代码任务（读写文件、执行命令、多轮工具调用等）。

目前已验证支持 Qwen3 系列模型（需模型支持工具调用格式）。

### 运行示例

```shell
# 指定任务描述，智能体将自动拆解并执行
cccc-windows --agent -c work\qwen3\qwen3_0.5b.ini --task "在当前目录下创建一个 hello.py，内容为打印 Hello World"
```

## 编译说明

详见 [doc/编译说明.md](doc/编译说明.md)。

### logo

logo由Dr. Cheng ZD <https://github.com/vorbei> 设计。

旧版logo由Mr. Zou YB设计。

