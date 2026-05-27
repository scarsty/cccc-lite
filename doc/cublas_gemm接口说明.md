# cublas_real.h — GEMM 接口说明

所有接口均定义在 `cccc/cublas_real.h` 的 `Cublas` 类中，列主序（column-major）约定，计算语义为：

$$C = \alpha \cdot \text{op}(A) \cdot \text{op}(B) + \beta \cdot C$$

---

## 一、标准精度 gemm（直接调用 cuBLAS）

| # | 签名（简化） | 底层 API | 用途 |
|---|------------|---------|------|
| 1 | `gemm(float* A, float* B, float* C)` | `cublasSgemm` | FP32 全精度，训练/通用 |
| 2 | `gemm(double* A, double* B, double* C)` | `cublasDgemm` | FP64 高精度（极少使用） |
| 3 | `gemm(half* A, half* B, half* C)` | `cublasGemmEx` FP16 IO / FP32 acc | FP16 推理（Ampere 以前） |
| 4 | `gemm(bfloat16* A, bfloat16* B, bfloat16* C)` | `cublasGemmEx` BF16 IO / FP32 acc | BF16 推理（LLM 主路径） |

---

## 二、量化 gemm（FP8 权重）

### 2-1. `gemm(uint8_t* A, bfloat16* B → bfloat16* C)` — W8A16 / W8A8（权重在 A）

权重 A（FP8 E4M3/E5M2/FP4）× 激活 B（BF16）→ 输出 C（BF16）

| 参数 | 含义 |
|------|------|
| `inv_scale` | 权重反量化系数（`quant_scale_` 的倒数）|
| `weight_type` | 权重类型：`FP8_E4M3` / `FP8_E5M2` / `FP4_E2M1` |
| `act_scale` | 激活静态量化 scale（HF input_scale）；== 0 时走动态 absmax |

**执行路径（优先级从高到低）：**

```
weight_type == FP8_E5M2  →  cublasLt E5M2×BF16 kernel
                           （失败则退回 E5M2→BF16 dequant + BF16 GEMM）

weight_type == FP8_E4M3 且 act_data_type_==FP8_E4M3（W8A8 模式）：
   act_scale > 0  →  BF16 act → FP8（静态 1/input_scale）
   act_scale == 0 →  BF16 act → FP8（动态 cuda_bf16_to_fp8e4m3_dynamic）
   →  cublasLt FP8×FP8→BF16
   →（失败）调用 gemm(uint8_t*, uint8_t*, bfloat16*)  [见 2-3]

weight_type == FP8_E4M3（W8A16 模式）：
   FP8 weight → BF16 dequant（cuda_convert）+ cublasGemmEx BF16×BF16

weight_type == FP4_E2M1：
   FP4 weight → BF16 dequant（cuda_convert）+ cublasGemmEx BF16×BF16
```

---

### 2-2. `gemm(bfloat16* A, uint8_t* B → bfloat16* C)` — W8A16 / W8A8（权重在 B）

与 2-1 对称，激活 A（BF16）× 权重 B（FP8/FP4）→ 输出 C（BF16）。

执行路径与 2-1 完全对称，区别在于：量化激活是 A，权重反量化系数用于 B。

---

### 2-3. `gemm(uint8_t* A, uint8_t* B → bfloat16* C)` — FP8×FP8 → BF16（SGEMM 回退）

A（FP8）× B（FP8）→ C（BF16）；这是 2-1/2-2 的 W8A8 回退路径，也可直接调用。

```
FP8 A  → float（inv_scale_A）
FP8 B  → float（inv_scale_B）
cublasSgemm float×float → float
float C → BF16
```

> 注：当前此路径在 RTX 5090 实测中未被触发——cublasLt FP8 GEMM 直接成功。

---

### 2-4. `gemm(uint8_t* A, uint8_t* B → uint8_t* C_fp8)` — FP8×FP8 → FP8

A（FP8）× B（FP8）→ C（FP8）；供 W8A8 全路径输出也是 FP8 的场景使用（如 FFN 中间层直接保持 FP8）。

```
尝试 cublasLt FP8×FP8→BF16（lt_quant_gemm）
（失败）FP8→float→SGEMM→float→BF16
BF16 中间结果 → FP8（scale=1.0，last_fp8_out_scale_=1.0）
```

---

## 三、统一分派器 gemm（type-erased）

```cpp
gemm(TransA, TransB, M, N, K, alpha,
     const void* A, DataType typeA, lda, invScaleA,
     const void* B, DataType typeB, ldb, invScaleB,
     beta, void* C, DataType typeC, ldc, act_scale)
```

根据 `typeA`、`typeB`、`typeC` 自动分派到上述各具体重载：

| typeA | typeB | typeC | 分派目标 |
|-------|-------|-------|---------|
| 量化 | 量化 | FP8 | 2-4（FP8→FP8） |
| 量化 | 量化 | BF16 | 2-3（FP8×FP8→BF16） |
| 量化 | 非量化 | FP8 | 2-1→BF16 中间结果再转 FP8 |
| 量化 | 非量化 | BF16 | 2-1 |
| 非量化 | 量化 | FP8 | 2-2→BF16 中间结果再转 FP8 |
| 非量化 | 量化 | BF16 | 2-2 |
| 非量化 | 非量化 | FP64 | gemm double |
| 非量化 | 非量化 | 其他 | cublasGemmEx（FP16/BF16/FP32） |

---

## 四、批处理 gemmStridedBatched

| # | 类型 | 底层 API | 用途 |
|---|------|---------|------|
| 1 | FP32 | `cublasSgemmStridedBatched` | MHA 多头注意力（float 训练） |
| 2 | FP16 | `cublasGemmStridedBatchedEx` BF16 / FP32 acc | MHA（half 推理） |
| 3 | BF16 | `cublasGemmStridedBatchedEx` BF16 / FP32 acc | MHA（BF16 推理，LLM 主路径） |

---

## 五、内部辅助 lt_quant_gemm

```cpp
bool lt_quant_gemm(transA, transB, M, N, K, alpha,
                   A, lda, typeA, B, ldb, typeB,
                   beta, bfloat16* C, ldc)
```

使用 `cublasLtMatmul`（`CUBLAS_COMPUTE_32F`）对任意量化类型对做 GEMM。  
调用方在调用前须将 dequant scale 写入 `d_scale_a_`（设备内存）、`d_scale_b_`（量化 B 时）。  
返回 `true` 表示成功；`false` 时调用方负责执行回退路径。

---

## 六、总览（按调用频率）

```
推理 LLM（BF16 主路径）：
  prefill/decode GEMM  →  gemm(bfloat16)
  MHA SDPA            →  gemmStridedBatched(bfloat16)

推理 LLM（W8A16，FP8 权重 + BF16 激活）：
  所有权重 GEMM        →  gemm(uint8_t*, bfloat16*) / gemm(bfloat16*, uint8_t*)
                           内部：FP8→BF16 dequant + cublasGemmEx

推理 LLM（W8A8，FP8 权重 + FP8 激活）：
  所有权重 GEMM        →  同上但先量化激活为 FP8
                           内部：cublasLt FP8×FP8（RTX 5090 成功）
                           回退：gemm(uint8_t*, uint8_t*, bfloat16*)

训练：
  前向/反向 GEMM       →  gemm(float) / gemm(bfloat16)
```
