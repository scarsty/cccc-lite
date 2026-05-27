#!/usr/bin/env python3
"""
convert_weights.py — cccc 权重转换工具（统一版）

将 HuggingFace safetensors 格式的模型权重转换为 cccc INIReaderBin 格式。
同时可生成对应的 INI 配置文件（--gen_ini 或转换时自动生成）。

支持的模型类型（--model）：
  qwen3      Qwen3 系列（0.5B~72B），FP16 输出，从 config.json 自动读取结构参数
  hy-mt2     Hy-MT2-7B（腾讯混元翻译），官方 FP8-E4M3 量化格式，直接保留 FP8 字节
  llama-fp4  Llama-3.1-8B-Instruct-FP4（NVIDIA NVFP4），转换为 block-scale FP4-E2M1

输出格式（INIReaderBin）：
  32 字节头 "CFG_BIN INI" + uint64 INI 文本长度 + INI 索引文本 + 二进制数据块

用法：
  python convert_weights.py --model qwen3     --hf_dir <HF目录> [--output qwen3_cccc.bin]
  python convert_weights.py --model hy-mt2    --hf_dir <HF目录> [--output hy_mt2_cccc.bin]
  python convert_weights.py --model llama-fp4 --hf_dir <HF目录> [--output llama_fp4_cccc.bin]
  python convert_weights.py --model llama-fp4 --hf_dir <HF目录> --dry_run
  python convert_weights.py --model qwen3     --hf_dir <HF目录> --gen_ini  # 仅生成 INI

依赖：
  pip install safetensors torch numpy
"""

import argparse
import glob
import json
import os
import struct
import sys

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: torch 未安装，BF16 权重将使用 numpy 回退路径。")

try:
    from safetensors import safe_open
except ImportError:
    print("ERROR: safetensors 未安装。请运行: pip install safetensors")
    sys.exit(1)


# ── INIReaderBin 序列化 ───────────────────────────────────────────────────────


class BinWriter:
    """cccc INIReaderBin 格式写入器。"""

    def __init__(self):
        self.entries: dict[str, bytes] = {}

    def add(self, key: str, data):
        if isinstance(data, np.ndarray):
            self.entries[key] = data.tobytes()
        else:
            self.entries[key] = bytes(data)

    def save(self, filename: str):
        header = b"CFG_BIN INI" + b"\x00" * (32 - len("CFG_BIN INI"))
        offset = 0
        lines = []
        data_parts = []
        for key, val in self.entries.items():
            lines.append(f"{key} = {offset},{len(val)}")
            data_parts.append(val)
            offset += len(val)
        ini_bytes = ("\n".join(lines) + "\n").encode("ascii")
        data = b"".join(data_parts)
        with open(filename, "wb") as f:
            f.write(header)
            f.write(struct.pack("<Q", len(ini_bytes)))
            f.write(ini_bytes)
            f.write(data)
        total = 32 + 8 + len(ini_bytes) + len(data)
        print(f"已保存 {filename}：{total / 1024 / 1024 / 1024:.3f} GB，{len(self.entries)} 个条目")


# ── safetensors 辅助 ──────────────────────────────────────────────────────────


def open_safetensors(hf_dir: str) -> tuple[dict, dict]:
    """打开目录下所有 safetensors 分片，返回 (weight_index, sf_handles)。"""
    st_files = sorted(glob.glob(os.path.join(hf_dir, "*.safetensors")))
    if not st_files:
        print(f"ERROR: 在 {hf_dir} 中未找到 .safetensors 文件")
        sys.exit(1)
    print(f"找到 {len(st_files)} 个 safetensors 分片")
    framework = "pt" if HAS_TORCH else "numpy"
    weight_index: dict[str, str] = {}
    sf_handles: dict[str, object] = {}
    for st_file in st_files:
        h = safe_open(st_file, framework=framework, device="cpu")
        sf_handles[st_file] = h
        for key in h.keys():
            weight_index[key] = st_file
    print(f"共 {len(weight_index)} 个权重张量")
    return weight_index, sf_handles


def get_tensor(sf_handles: dict, weight_index: dict, name: str) -> "torch.Tensor":
    st_file = weight_index.get(name)
    if st_file is None:
        raise KeyError(f"权重未找到: {name}")
    return sf_handles[st_file].get_tensor(name)


def tensor_to_fp16_flat(t) -> np.ndarray:
    """任意 dtype tensor → float16 flat numpy 数组。"""
    if HAS_TORCH:
        return t.float().half().numpy().flatten()
    # numpy 回退
    if hasattr(t, "numpy"):
        t = t.numpy()
    if t.dtype == np.float16:
        return t.flatten()
    if t.dtype == np.float32:
        return t.astype(np.float16).flatten()
    # BF16 (2 字节) → FP32 → FP16
    u16 = np.frombuffer(t.tobytes(), dtype=np.uint16)[: t.size]
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    return f32.astype(np.float16)


def tensor_to_bf16_bytes(t) -> bytes:
    """任意 dtype tensor → bfloat16 raw bytes。"""
    if HAS_TORCH:
        return t.to(torch.bfloat16).view(torch.int16).numpy().tobytes()
    raise RuntimeError("tensor_to_bf16_bytes 需要 torch")


# ── 模型结构通用权重名映射 ────────────────────────────────────────────────────


def build_weight_map_qwen3(num_layers: int, has_qknorm: bool = True):
    """Qwen3 系列权重映射（HF name → cccc name, needs_transpose, is_quantized）。"""
    mapping = [
        ("model.embed_tokens.weight", "W_emb", False, False),
        ("model.norm.weight", "W_rms_final", False, False),
    ]
    for i in range(num_layers):
        pfx = f"model.layers.{i}"
        mapping += [
            (f"{pfx}.input_layernorm.weight", f"W_rms_attn_{i}", False, False),
            (f"{pfx}.self_attn.q_proj.weight", f"W_q_{i}", False, False),
            (f"{pfx}.self_attn.k_proj.weight", f"W_k_{i}", False, False),
            (f"{pfx}.self_attn.v_proj.weight", f"W_v_{i}", False, False),
            (f"{pfx}.self_attn.o_proj.weight", f"W_o_{i}", False, False),
            (f"{pfx}.post_attention_layernorm.weight", f"W_rms_ffn_{i}", False, False),
            (f"{pfx}.mlp.gate_proj.weight", f"W_gate_{i}", False, False),
            (f"{pfx}.mlp.up_proj.weight", f"W_up_{i}", False, False),
            (f"{pfx}.mlp.down_proj.weight", f"W_down_{i}", False, False),
        ]
        if has_qknorm:
            mapping += [
                (f"{pfx}.self_attn.q_norm.weight", f"W_qnorm_{i}", False, False),
                (f"{pfx}.self_attn.k_norm.weight", f"W_knorm_{i}", False, False),
            ]
    return mapping


def build_weight_map_hy_mt2(num_layers: int = 32):
    """Hy-MT2-7B 权重映射（大权重 FP8，小权重 BF16）。"""
    mapping = [
        ("model.embed_tokens.weight", "W_emb", False, False),
        ("model.norm.weight", "W_rms_final", False, False),
    ]
    for i in range(num_layers):
        pfx = f"model.layers.{i}"
        mapping += [
            # 量化大权重
            (f"{pfx}.self_attn.q_proj.weight", f"W_q_{i}", False, True),
            (f"{pfx}.self_attn.k_proj.weight", f"W_k_{i}", False, True),
            (f"{pfx}.self_attn.v_proj.weight", f"W_v_{i}", False, True),
            (f"{pfx}.self_attn.o_proj.weight", f"W_o_{i}", False, True),
            (f"{pfx}.mlp.gate_proj.weight", f"W_gate_{i}", False, True),
            (f"{pfx}.mlp.up_proj.weight", f"W_up_{i}", False, True),
            (f"{pfx}.mlp.down_proj.weight", f"W_down_{i}", False, True),
            # 非量化小权重
            (f"{pfx}.input_layernorm.weight", f"W_rms_attn_{i}", False, False),
            (f"{pfx}.post_attention_layernorm.weight", f"W_rms_ffn_{i}", False, False),
            (f"{pfx}.self_attn.query_layernorm.weight", f"W_qnorm_{i}", False, False),
            (f"{pfx}.self_attn.key_layernorm.weight", f"W_knorm_{i}", False, False),
        ]
    return mapping


def build_weight_map_llama(num_layers: int = 32):
    """Llama 3.x 权重映射（大权重 NVFP4，小权重 BF16，无 qknorm）。"""
    mapping = [
        ("model.embed_tokens.weight", "W_emb", False, False),
        ("model.norm.weight", "W_rms_final", False, False),
        ("lm_head.weight", "W_lm_head", False, False),
    ]
    for i in range(num_layers):
        pfx = f"model.layers.{i}"
        mapping += [
            (f"{pfx}.self_attn.q_proj.weight", f"W_q_{i}", False, True),
            (f"{pfx}.self_attn.k_proj.weight", f"W_k_{i}", False, True),
            (f"{pfx}.self_attn.v_proj.weight", f"W_v_{i}", False, True),
            (f"{pfx}.self_attn.o_proj.weight", f"W_o_{i}", False, True),
            (f"{pfx}.mlp.gate_proj.weight", f"W_gate_{i}", False, True),
            (f"{pfx}.mlp.up_proj.weight", f"W_up_{i}", False, True),
            (f"{pfx}.mlp.down_proj.weight", f"W_down_{i}", False, True),
            (f"{pfx}.input_layernorm.weight", f"W_rms_attn_{i}", False, False),
            (f"{pfx}.post_attention_layernorm.weight", f"W_rms_ffn_{i}", False, False),
        ]
    return mapping


# ── Qwen3 转换（FP16） ────────────────────────────────────────────────────────


def convert_qwen3(hf_dir: str, output_path: str):
    cfg_path = os.path.join(hf_dir, "config.json")
    if not os.path.exists(cfg_path):
        print(f"ERROR: config.json 未找到: {cfg_path}")
        sys.exit(1)
    with open(cfg_path) as f:
        cfg = json.load(f)

    LAYERS = cfg["num_hidden_layers"]
    tied = cfg.get("tie_word_embeddings", False)
    has_qknorm = any(
        k in cfg for k in ("qk_norm", "use_qk_norm")
    ) or cfg.get("model_type", "") == "qwen3"
    print(
        f"Qwen3  layers={LAYERS}  tie_embeddings={tied}  qknorm={has_qknorm}"
    )

    weight_index, sf_handles = open_safetensors(hf_dir)
    mapping = build_weight_map_qwen3(LAYERS, has_qknorm=has_qknorm)

    writer = BinWriter()
    writer.add("data_type", b"half")
    writer.add("named_weights", b"1")

    W_emb = None
    for hf_name, cccc_name, _, _ in mapping:
        if hf_name not in weight_index:
            print(f"  MISS  {hf_name}")
            continue
        print(f"  加载 {hf_name} ...", end="\r", flush=True)
        t = get_tensor(sf_handles, weight_index, hf_name)
        arr = tensor_to_fp16_flat(t)
        writer.add(f"weight_{cccc_name}", arr)
        if cccc_name == "W_emb":
            W_emb = arr

    # tied lm_head
    if tied and W_emb is not None and "lm_head.weight" not in weight_index:
        print(f"\n  tie_word_embeddings=True，W_lm_head 复用 W_emb")
        writer.add("weight_W_lm_head", W_emb)
    elif "lm_head.weight" in weight_index:
        t = get_tensor(sf_handles, weight_index, "lm_head.weight")
        writer.add("weight_W_lm_head", tensor_to_fp16_flat(t))

    print(f"\n写入 {output_path} ...")
    writer.save(output_path)


# ── Hy-MT2 转换（FP8-E4M3） ───────────────────────────────────────────────────


def convert_hy_mt2(hf_dir: str, output_path: str):
    if not HAS_TORCH:
        print("ERROR: hy-mt2 转换需要 torch")
        sys.exit(1)

    weight_index, sf_handles = open_safetensors(hf_dir)
    mapping = build_weight_map_hy_mt2(num_layers=32)

    writer = BinWriter()
    writer.add("data_type", b"fp8_e4m3")
    writer.add("named_weights", b"1")

    for hf_name, cccc_name, _, is_quantized in mapping:
        if hf_name not in weight_index:
            print(f"  MISS  {hf_name}")
            continue

        t = get_tensor(sf_handles, weight_index, hf_name)

        if is_quantized:
            # FP8 原始字节
            scale_key = hf_name + "_scale"
            if scale_key in weight_index:
                scale_val = float(get_tensor(sf_handles, weight_index, scale_key).reshape(-1)[0].item())
            else:
                print(f"  WARNING: {hf_name} 无 scale，使用 1.0")
                scale_val = 1.0
            input_scale_key = hf_name.replace(".weight", ".input_scale")
            fp8_blob = t.view(torch.uint8).numpy().tobytes()
            writer.add(f"weight_fp8_e4m3_{cccc_name}", fp8_blob)
            writer.add(f"scale_{cccc_name}", struct.pack("<f", scale_val))
            if input_scale_key in weight_index:
                ist = get_tensor(sf_handles, weight_index, input_scale_key)
                writer.add(f"input_scale_{cccc_name}", struct.pack("<f", float(ist.reshape(-1)[0].item())))
            print(f"  FP8  {hf_name:55s} → {cccc_name}  scale={scale_val:.4g}")
        else:
            blob = tensor_to_bf16_bytes(t)
            writer.add(f"weight_bf16_{cccc_name}", blob)
            print(f"  BF16 {hf_name:55s} → {cccc_name}")

    print(f"\n写入 {output_path} ...")
    writer.save(output_path)


# ── Llama FP4 转换（NVFP4 → block-scale FP4-E2M1） ───────────────────────────

# FP4 E2M1 查值表（16 个合法值）
_FP4_TABLE = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)
_FP4_KMAX = 6.0
_FP4_BOUNDARIES = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)


def _fp8_e4m3_to_float32(t: "torch.Tensor") -> np.ndarray:
    return t.to(torch.float32).numpy()


def _dequant_nvfp4(packed_t, scale_t, scale2: float, in_features: int, block_size: int = 16) -> np.ndarray:
    """NVFP4 → float32 flat（全向量化）。"""
    packed = packed_t.view(torch.uint8).numpy()
    out_features = packed.shape[0]
    scale_np = _fp8_e4m3_to_float32(scale_t).reshape(out_features, -1) * scale2
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    result = np.empty((out_features, in_features), dtype=np.float32)
    result[:, 0::2] = _FP4_TABLE[lo]
    result[:, 1::2] = _FP4_TABLE[hi]
    scale_expanded = np.repeat(scale_np, block_size, axis=1)[:, :in_features]
    result *= scale_expanded
    return result.ravel()


def _requant_fp4_blockscale(data: np.ndarray, group_size: int = 16):
    """float32 → block-scale FP4 packed nibbles + FP8-E4M3 block scales。"""
    n = len(data)
    pad = (-n) % group_size
    if pad:
        data = np.concatenate([data, np.zeros(pad, dtype=np.float32)])
    n_groups = len(data) // group_size
    data_2d = data.reshape(n_groups, group_size)
    absmax = np.abs(data_2d).max(axis=1)
    scales = np.where(absmax > 1e-10, absmax / _FP4_KMAX, 1.0).astype(np.float32)
    fwd = np.where(absmax > 1e-10, _FP4_KMAX / absmax, 1.0)
    scaled = (data_2d * fwd[:, np.newaxis]).ravel()

    sign_bits = (scaled < 0).astype(np.uint8) * 8
    pos_codes = np.searchsorted(_FP4_BOUNDARIES, np.abs(scaled)).astype(np.uint8)
    nibbles_flat = pos_codes | sign_bits

    packed = (nibbles_flat[0::2] | (nibbles_flat[1::2] << 4)).astype(np.uint8)
    n_bytes = (n + 1) // 2
    packed = packed[:n_bytes]

    n_valid_groups = (n + group_size - 1) // group_size
    scales = scales[:n_valid_groups]
    scales_fp8 = torch.from_numpy(scales).to(torch.float8_e4m3fn)
    return packed.tobytes(), scales_fp8.view(torch.uint8).numpy().tobytes()


def convert_llama_fp4(hf_dir: str, output_path: str, group_size: int = 16, dry_run: bool = False):
    if not HAS_TORCH:
        print("ERROR: llama-fp4 转换需要 torch")
        sys.exit(1)

    weight_index, sf_handles = open_safetensors(hf_dir)

    if dry_run:
        print("\n── 所有 key ──")
        for k in sorted(weight_index):
            t = get_tensor(sf_handles, weight_index, k)
            print(f"  {k}: {t.dtype}  {list(t.shape)}")
        return

    mapping = build_weight_map_llama(num_layers=32)

    writer = BinWriter()
    writer.add("data_type", b"fp4_e2m1")
    writer.add("named_weights", b"1")

    def _get_scale_key(wk):
        for sfx in (".weight_scale_inv", ".weight_scale"):
            k = wk.replace(".weight", sfx)
            if k in weight_index:
                return k
        return None

    for hf_name, cccc_name, _, is_quantized in mapping:
        if hf_name not in weight_index:
            print(f"  MISS  {hf_name}")
            continue
        t = get_tensor(sf_handles, weight_index, hf_name)

        if is_quantized:
            scale_key = _get_scale_key(hf_name)
            if scale_key is None:
                # 回退 BF16
                blob = tensor_to_bf16_bytes(t)
                writer.add(f"weight_bf16_{cccc_name}", blob)
                print(f"  BF16(fallback) {hf_name}")
                continue

            scale_t = get_tensor(sf_handles, weight_index, scale_key)
            scale2_key = hf_name.replace(".weight", ".weight_scale_2")
            scale2 = float(get_tensor(sf_handles, weight_index, scale2_key).item()) if scale2_key in weight_index else 1.0

            packed_np = t.view(torch.uint8).numpy()
            in_features = packed_np.shape[1] * 2
            data_f32 = _dequant_nvfp4(t, scale_t, scale2, in_features)
            nibbles_bytes, scales_bytes = _requant_fp4_blockscale(data_f32, group_size)

            writer.add(f"weight_fp4_e2m1_{cccc_name}", nibbles_bytes)
            writer.add(f"blockscale_{cccc_name}", scales_bytes)
            print(
                f"  FP4  {hf_name:55s} → {cccc_name}  "
                f"nibbles={len(nibbles_bytes)//1024}KB  scales={len(scales_bytes)//1024}KB"
            )
        else:
            blob = tensor_to_bf16_bytes(t)
            writer.add(f"weight_bf16_{cccc_name}", blob)
            print(f"  BF16 {hf_name:55s} → {cccc_name}")

    print(f"\n写入 {output_path} ...")
    writer.save(output_path)


# ── 入口 ──────────────────────────────────────────────────────────────────────


# ── INI / cifa 结构生成器 ─────────────────────────────────────────────────────

# 每种模型的[llm]默认值
_LLM_DEFAULTS = {
    "qwen3": dict(
        tokenizer="tokenizer.json",
        eos_tokens="151645",
        think_open_id="151668",
        think_close_id="151669",
        no_think_str="<|im_end|>",
        sys_prefix="<|im_start|>system\\n",
        sys_suffix="<|im_end|>\\n",
        user_prefix="<|im_start|>user\\n",
        user_suffix="<|im_end|>\\n",
        asst_prefix="<|im_start|>assistant\\n",
        asst_suffix="",
    ),
    "llama": dict(
        tokenizer="tokenizer.json",
        eos_tokens="128001,128009",
        think_open_id="-1",
        think_close_id="-1",
        no_think_str="",
        sys_prefix="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n",
        sys_suffix="<|eot_id|>",
        user_prefix="<|start_header_id|>user<|end_header_id|>\\n\\n",
        user_suffix="<|eot_id|>",
        asst_prefix="<|start_header_id|>assistant<|end_header_id|>\\n\\n",
        asst_suffix="<|eot_id|>",
    ),
    "hy-mt2": dict(
        tokenizer="tokenizer.json",
        eos_tokens="127960",
        think_open_id="-1",
        think_close_id="-1",
        no_think_str="",
        sys_prefix="<|startoftext|>",
        sys_suffix="<|extra_4|>",
        user_prefix="<|startoftext|>",
        user_suffix="<|extra_0|>",
        asst_prefix="",
        asst_suffix="<|eos|>",
    ),
}


def _gen_attn_block(has_qknorm: bool, groups: int) -> str:
    """生成单层 attention 块 cifa 代码（不含前后 RMSNorm）。"""
    qk_rope_q = """\
        W_q = MatrixWithName("W_q_" + to_string(i), D, Dq, 1, 1);
        Q_dq = batchedMul(W_q, X_norm, 1, 0);
        Q_hHT = reshape(Q_dq, {hd, H, T, 1});"""
    if has_qknorm:
        qk_rope_q += """
        W_qnorm = MatrixWithName("W_qnorm_" + to_string(i), hd, 1);
        Q_qnorm = rmsNorm(Q_hHT, W_qnorm);
        Q_hTH = permute(Q_qnorm, {0, 2, 1, 3});"""
    else:
        qk_rope_q += """
        Q_hTH = permute(Q_hHT, {0, 2, 1, 3});"""
    qk_rope_q += """
        Q_hb = reshapeBatch(Q_hTH, {hd, T, 1, HB});
        Q_r = rope(Q_hb, cos_tab, sin_tab);"""

    qk_rope_k = """\
        W_k = MatrixWithName("W_k_" + to_string(i), D, Dkv, 1, 1);
        K_dkv = batchedMul(W_k, X_norm, 1, 0);
        K_hHkvT = reshape(K_dkv, {hd, Hkv, T, 1});"""
    if has_qknorm:
        qk_rope_k += """
        W_knorm = MatrixWithName("W_knorm_" + to_string(i), hd, 1);
        K_knorm = rmsNorm(K_hHkvT, W_knorm);
        K_hTHkv = permute(K_knorm, {0, 2, 1, 3});"""
    else:
        qk_rope_k += """
        K_hTHkv = permute(K_hHkvT, {0, 2, 1, 3});"""
    qk_rope_k += """
        K_hb = reshapeBatch(K_hTHkv, {hd, T, 1, HkvB});
        K_r = rope(K_hb, cos_tab, sin_tab);"""

    v_block = """\
        W_v = MatrixWithName("W_v_" + to_string(i), D, Dkv, 1, 1);
        V_dkv = batchedMul(W_v, X_norm, 1, 0);
        V_hHkvT = reshape(V_dkv, {hd, Hkv, T, 1});
        V_hTHkv = permute(V_hHkvT, {0, 2, 1, 3});
        V_hb = reshapeBatch(V_hTHkv, {hd, T, 1, HkvB});"""

    kvcache_block = """\
        Kcache = MatrixWithName("Kcache_" + to_string(i), hd, T, 1, HkvB);
        setIsWeight(Kcache, 0);
        registerMatrix("Kcache_" + to_string(i), Kcache);
        K_cached = kvcache(K_r, Kcache);

        Vcache = MatrixWithName("Vcache_" + to_string(i), hd, T, 1, HkvB);
        setIsWeight(Vcache, 0);
        registerMatrix("Vcache_" + to_string(i), Vcache);
        V_cached = kvcache(V_hb, Vcache);"""

    if groups == 1:
        # MHA：无需 tile
        gqa_block = """\
        K_tiled = K_cached;
        V_tiled = V_cached;"""
    else:
        gqa_block = f"""\
        K_r2   = reshapeBatch(K_cached,  {{hd, T, HkvB, 1}});
        K_r3   = tile(K_r2, {{1, 1, 1, {groups}}});
        K_r4   = permute(K_r3, {{0, 1, 3, 2}});
        K_tiled = reshapeBatch(K_r4, {{hd, T, 1, HB}});

        V_r2   = reshapeBatch(V_cached, {{hd, T, HkvB, 1}});
        V_r3   = tile(V_r2, {{1, 1, 1, {groups}}});
        V_r4   = permute(V_r3, {{0, 1, 3, 2}});
        V_tiled = reshapeBatch(V_r4, {{hd, T, 1, HB}});"""

    attn_out = """\
        Attn = attention(Q_r, K_tiled, V_tiled, hd, 1);

        Attn_hTH = reshapeBatch(Attn, {hd, T, H, 1});
        Attn_hHT = permute(Attn_hTH, {0, 2, 1, 3});
        Attn_flat = reshape(Attn_hHT, {Dq, T, 1, 1});
        W_o = MatrixWithName("W_o_" + to_string(i), Dq, D, 1, 1);
        O_out = batchedMul(W_o, Attn_flat, 1, 0);

        R1 = X + O_out;"""

    return "\n".join([qk_rope_q, qk_rope_k, v_block, kvcache_block, gqa_block, attn_out])


def _gen_ffn_block() -> str:
    return """\
        W_rms_ffn = MatrixWithName("W_rms_ffn_" + to_string(i), D, 1);
        R1_norm = rmsNorm(R1, W_rms_ffn);

        W_gate = MatrixWithName("W_gate_" + to_string(i), D, I, 1, 1);
        W_up   = MatrixWithName("W_up_" + to_string(i), D, I, 1, 1);
        gate_out = silu(batchedMul(W_gate, R1_norm, 1, 0));
        up_out   = batchedMul(W_up, R1_norm, 1, 0);
        gated    = elementMul(gate_out, up_out);
        W_down = MatrixWithName("W_down_" + to_string(i), I, D, 1, 1);
        ffn_out = batchedMul(W_down, gated, 1, 0);

        X = R1 + ffn_out;"""


def generate_ini(
    model_type: str,           # "qwen3" / "llama" / "hy-mt2"
    bin_file: str,             # 相对/绝对路径的 .bin 文件名
    D: int,                    # hidden_size
    H: int,                    # num_attention_heads
    Hkv: int,                  # num_key_value_heads
    hd: int,                   # head_dim
    I: int,                    # intermediate_size
    V: int,                    # vocab_size
    LAYERS: int,               # num_hidden_layers
    rope_theta: float,
    has_qknorm: bool,
    tie_word_embeddings: bool,
    context_len: int = 1024,
    data_type: str = "half",
    weight_data_type: str = "",
    llm_overrides: dict = None,
) -> str:
    """生成完整 cccc INI 文件内容（str）。structure1 由 cccc 自动从 structure0 派生。"""

    Dq = H * hd
    Dkv = Hkv * hd
    groups = H // Hkv
    T = context_len
    HB = H
    HkvB = Hkv

    llm_d = dict(_LLM_DEFAULTS.get(model_type, _LLM_DEFAULTS["qwen3"]))
    if llm_overrides:
        llm_d.update(llm_overrides)

    # [train] section
    wdt_line = f"\nweight_data_type = {weight_data_type}" if weight_data_type else ""
    tie_line = "\ntie_word_embeddings = 1" if tie_word_embeddings else ""
    train_sec = f"""\
[train]
load_file = {bin_file}
data_type = {data_type}{wdt_line}{tie_line}
"""

    # [llm] section
    llm_sec = f"""\
[llm]
tokenizer = {llm_d['tokenizer']}
eos_tokens = {llm_d['eos_tokens']}
think_open_id = {llm_d['think_open_id']}
think_close_id = {llm_d['think_close_id']}
no_think_str = {llm_d['no_think_str']}
sys_prefix = {llm_d['sys_prefix']}
sys_suffix = {llm_d['sys_suffix']}
user_prefix = {llm_d['user_prefix']}
user_suffix = {llm_d['user_suffix']}
asst_prefix = {llm_d['asst_prefix']}
asst_suffix = {llm_d['asst_suffix']}

[net]
net_num = 2
"""

    # cifa structure0
    rope_theta_str = f"{rope_theta:.1f}" if rope_theta == int(rope_theta) else str(rope_theta)
    attn_block = _gen_attn_block(has_qknorm=has_qknorm, groups=groups)
    ffn_block = _gen_ffn_block()

    lm_head_code = ""
    if not tie_word_embeddings:
        lm_head_code = '\n    W_lm_head = MatrixWithName("W_lm_head", D, V, 1, 1);'

    structure0 = f"""\
structure0='
    D    = {D};
    Dq   = {Dq};
    Dkv  = {Dkv};
    H    = {H};
    Hkv  = {Hkv};
    hd   = {hd};
    I    = {I};
    V    = {V};
    T    = {T};
    HB   = {HB};
    HkvB = {HkvB};

    W_emb = MatrixWithName("W_emb", D, 1, 1, V);
    token_ids = MatrixF(T, 1, 1, 1);
    X = embed(token_ids, W_emb);

    cos_tab = ropeCosTbl(T, hd, {rope_theta_str});
    sin_tab = ropeSinTbl(T, hd, {rope_theta_str});

    for (i = 0; i < {LAYERS}; i++)
    {{
        W_rms_attn = MatrixWithName("W_rms_attn_" + to_string(i), D, 1);
        X_norm = rmsNorm(X, W_rms_attn);

{attn_block}

{ffn_block}
    }}

    W_rms_final = MatrixWithName("W_rms_final", D, 1);
    X_final = rmsNorm(X, W_rms_final);{lm_head_code}
    logits = batchedMul(W_lm_head, X_final, 1, 0);

    setXY(token_ids, logits);
'
"""

    return train_sec + llm_sec + structure0


def generate_ini_from_config(model_type: str, hf_dir: str, bin_file: str,
                              context_len: int = 1024,
                              data_type: str = None,
                              weight_data_type: str = "") -> str:
    """从 config.json 自动读取参数并生成 INI（str）。"""
    cfg_path = os.path.join(hf_dir, "config.json")
    if not os.path.exists(cfg_path):
        print(f"ERROR: config.json 未找到: {cfg_path}")
        sys.exit(1)
    with open(cfg_path) as f:
        cfg = json.load(f)

    D = cfg["hidden_size"]
    H = cfg["num_attention_heads"]
    Hkv = cfg.get("num_key_value_heads", H)
    hd = cfg.get("head_dim", D // H)
    I = cfg["intermediate_size"]
    V = cfg["vocab_size"]
    LAYERS = cfg["num_hidden_layers"]
    rope_theta = float(cfg.get("rope_theta", 10000.0))
    tie = cfg.get("tie_word_embeddings", False)
    arch = cfg.get("model_type", "")

    # qknorm 判断
    if model_type == "qwen3":
        has_qknorm = True
    elif model_type in ("hy-mt2",):
        has_qknorm = True
    else:
        has_qknorm = False

    if data_type is None:
        data_type = "half" if model_type == "qwen3" else "bfloat16"

    print(f"  config.json: D={D} H={H} Hkv={Hkv} hd={hd} I={I} V={V} LAYERS={LAYERS}")
    print(f"  rope_theta={rope_theta}  tie={tie}  qknorm={has_qknorm}")

    return generate_ini(
        model_type=model_type,
        bin_file=bin_file,
        D=D, H=H, Hkv=Hkv, hd=hd, I=I, V=V, LAYERS=LAYERS,
        rope_theta=rope_theta,
        has_qknorm=has_qknorm,
        tie_word_embeddings=tie,
        context_len=context_len,
        data_type=data_type,
        weight_data_type=weight_data_type,
    )


DEFAULTS = {
    "qwen3": "qwen3_cccc.bin",
    "hy-mt2": "hy_mt2_cccc.bin",
    "llama-fp4": "llama_fp4bs_cccc.bin",
}

INI_DEFAULTS = {
    "qwen3": "qwen3.ini",
    "hy-mt2": "hy_mt2.ini",
    "llama-fp4": "llama_fp4.ini",
}


def main():
    parser = argparse.ArgumentParser(
        description="cccc 权重转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["qwen3", "hy-mt2", "llama-fp4"],
        help="模型类型",
    )
    parser.add_argument(
        "--hf_dir",
        required=True,
        help="HuggingFace 模型目录（含 *.safetensors 和 config.json）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出 .bin 文件路径（默认在 hf_dir 下自动命名）",
    )
    parser.add_argument(
        "--gen_ini",
        action="store_true",
        help="仅生成 INI 配置文件，不转换权重",
    )
    parser.add_argument(
        "--ini_output",
        default=None,
        help="输出 .ini 文件路径（默认与 --output 同目录，同名但扩展名 .ini）",
    )
    parser.add_argument(
        "--context_len",
        type=int,
        default=1024,
        help="预填充上下文长度（默认 1024）",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=16,
        help="[llama-fp4] FP4 block size（默认 16）",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="[llama-fp4] 仅列出权重 key，不写文件",
    )
    args = parser.parse_args()

    hf_dir = os.path.abspath(args.hf_dir)
    if not os.path.isdir(hf_dir):
        print(f"ERROR: hf_dir 不存在: {hf_dir}")
        sys.exit(1)

    output = args.output
    if output is None:
        output = os.path.join(hf_dir, DEFAULTS[args.model])
    output = os.path.abspath(output)

    ini_output = args.ini_output
    if ini_output is None:
        base = os.path.splitext(output)[0]
        ini_output = base + ".ini"
    ini_output = os.path.abspath(ini_output)

    print(f"模型类型: {args.model}")
    print(f"输入目录: {hf_dir}")
    print(f"输出 bin: {output}")
    print(f"输出 ini: {ini_output}")
    print()

    # ── INI 生成 ──
    model_key = "llama" if args.model == "llama-fp4" else args.model
    wdt = "fp4_e2m1" if args.model == "llama-fp4" else (
          "fp8_e4m3" if args.model == "hy-mt2" else "")
    dt = "bfloat16" if args.model in ("llama-fp4", "hy-mt2") else "half"

    ini_content = generate_ini_from_config(
        model_type=model_key,
        hf_dir=hf_dir,
        bin_file=os.path.basename(output),
        context_len=args.context_len,
        data_type=dt,
        weight_data_type=wdt,
    )
    with open(ini_output, "w", encoding="utf-8") as f:
        f.write(ini_content)
    print(f"已生成 INI: {ini_output}")

    if args.gen_ini:
        print("\n完成（仅生成 INI）！")
        return

    print()
    if args.model == "qwen3":
        convert_qwen3(hf_dir, output)
    elif args.model == "hy-mt2":
        convert_hy_mt2(hf_dir, output)
    elif args.model == "llama-fp4":
        convert_llama_fp4(hf_dir, output, group_size=args.group_size, dry_run=args.dry_run)

    print("\n完成！")


if __name__ == "__main__":
    main()
