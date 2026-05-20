# AMD ROCm / HIP 相关说明

## 1. RX 7900 XTX 深度学习性能详细分析

### 1.1 硬件规格

| 指标 | RX 7900 XTX | RTX 4080 | RTX 4090 |
|------|------------|---------|---------|
| 架构 | RDNA3 (gfx1100) | Ada Lovelace | Ada Lovelace |
| 计算单元/SM 数 | 96 CU | 76 SM | 128 SM |
| FP32 算力 | **61.4 TFLOPS** | 48.7 TFLOPS | 82.6 TFLOPS |
| FP16 着色器 | **122.8 TFLOPS** | 97.4 TFLOPS | 165.2 TFLOPS |
| FP16 Tensor Core | ❌ 无 | **390 TFLOPS** | **660 TFLOPS** |
| BF16 Tensor Core | ❌ 无 | **390 TFLOPS** | **660 TFLOPS** |
| FP8 Tensor Core | ❌ 无 | **780 TFLOPS** | **1321 TFLOPS** |
| 显存 | 24 GB GDDR6 | 16 GB GDDR6X | 24 GB GDDR6X |
| 显存带宽 | **960 GB/s** | 717 GB/s | 1008 GB/s |
| L2 缓存 | ~6 MB | 64 MB | **72 MB** |
| TDP | 355 W | 320 W | 450 W |

### 1.2 为什么差距如此巨大：Tensor Core 的作用

NVIDIA 自 Volta 架构引入 **Tensor Core**，是专用的矩阵乘加单元（MMA），可在单个时钟周期内完成一整块矩阵运算：

- 一个 Tensor Core 每周期可完成 4×4×4 = 64 次 FP16 乘加，等效 128 次浮点运算
- 普通 CUDA Core 每周期仅能完成 2 次浮点运算（FMA）
- RTX 4090 的 FP16 Tensor 算力（660 TFLOPS）是 FP32 算力（83 TFLOPS）的约 **8×**

**AMD RDNA3 没有等效的专用 MMA 单元**，FP16 仅比 FP32 快 2×（SIMD 打包执行），而非 8×。

### 1.3 大 L2 缓存对 Transformer 的影响

Ada Lovelace 架构（RTX 40 系）拥有 **72 MB L2 缓存**，对 Transformer 模型影响显著：

- BERT-Large 激活数据（序列长度 512）约 50-60 MB，可完全放入 L2
- 大批量训练时，注意力机制的 K/V 矩阵可以多次复用而无需从显存重新加载
- 这在 Transformer 架构上可带来额外 **1.5–2×** 的实际加速

RX 7900 XTX 的 L2 缓存约 6 MB，无法实现同样效果。

### 1.4 实际深度学习性能对比（参考数据）

| 任务 | 批次 | RX 7900 XTX | RTX 4080 | RTX 4090 | 备注 |
|------|------|------------|---------|---------|------|
| ResNet-50 训练（img/s） | 256 | ~1,800 | ~4,500 | ~7,000 | FP16 混合精度 |
| BERT-Large 训练（seq/s） | 32 | ~380 | ~1,100 | ~1,800 | FP16，受 L2 缓存影响 |
| LLM 推理 token/s（Llama-2-7B）| batch=1 | ~65 | ~55 | **~100** | 带宽限制，7900XT竞争力强 |
| LLM 推理 token/s（Llama-2-7B）| batch=16 | ~220 | ~450 | ~700 | 计算限制，Tensor Core 差距显现 |
| Stable Diffusion（it/s）| 512×512 | ~7 | ~20 | ~33 | FP16，卷积密集 |

> **说明**：以上数据为参考量级，实际表现取决于驱动版本、ROCm/CUDA 版本和具体模型实现。

### 1.5 训练 vs 推理分场景分析

**训练场景**（计算密集）：
- 大批量、反复矩阵乘法，Tensor Core 优势完全发挥
- RTX 4080 约 **2.5–3×** 领先 7900 XTX
- RTX 4090 约 **4–5×** 领先 7900 XTX
- 适用：ResNet/ViT 图像分类，BERT/GPT 语言模型全量训练

**推理场景·大批量**（计算密集）：
- 同上，Tensor Core 差距仍在，且 FP8 支持对 RTX 40 系额外加成
- 数据中心量化推理场景中 NVIDIA 优势更大

**推理场景·小批量 / LLM token 生成**（带宽限制）：
- 单次生成时权重从显存逐层加载，带宽成为瓶颈
- 7900 XTX（960 GB/s）≈ RTX 4090（1008 GB/s），**远超** RTX 4080（717 GB/s）
- 小批量 LLM 推理中 7900 XTX 与 RTX 4090 **性能相近**，是其最具竞争力的场景

### 1.6 综合结论

| 场景 | 7900 XTX vs RTX 4080 | 7900 XTX vs RTX 4090 |
|------|---------------------|---------------------|
| 深度学习训练 | 落后约 2.5–3× | 落后约 4–5× |
| 大批量推理 | 落后约 2–3× | 落后约 3–4× |
| LLM 小批量推理 | **领先约 30%** | 基本持平 |
| 显存容量 | **优势（24 GB vs 16 GB）** | 相同（均 24 GB）|

AMD 7900 XTX 在训练任务上与 NVIDIA 差距根本原因是缺乏 Tensor Core，而非 FLOPS 不够。在带宽密集型的 LLM 小批量推理中，其 960 GB/s 带宽带来相当竞争力。

---

## 2. AMD GPU 计算后端性能对比

适用于深度学习（推理）的四种 AMD GPU 计算后端，以 RX 7900 XTX 为基准：

| 后端 | 平台 | 训练 | 推理性能（参考基准）| 安装复杂度 |
|------|------|------|-------------------|-----------|
| **ROCm/HIP** | Linux（Windows 实验性）| ✅ 完整 | **100%**（基准） | 中等 |
| **Vulkan Compute** | 全平台 | ❌ 基本无 | ~75–85% | 简单 |
| **DirectML** | Windows 专属 | ❌ 无 | ~60–75% | 简单 |
| **OpenCL** | 全平台 | ❌ 极少 | ~50–65% | 简单 |

- **ROCm**：最佳性能，MIOpen/rocBLAS 高度优化，Linux 独占完整支持
- **Vulkan**：跨平台首选（llama.cpp ggml-vulkan 后端），Windows 可用，无需特殊驱动
- **DirectML**：Windows 推理场景，通过 ONNX Runtime 使用，官方维护稳定
- **OpenCL**：历史遗留，不建议新项目使用

---

## 3. ROCm / HIP SDK：Windows 与 Linux 官方组件对比

来源：AMD 官方文档 ROCm 7.2.3（2026-02-19）

| 组件 | Linux (ROCm) | Windows (HIP SDK) |
|------|-------------|-------------------|
| HIP 运行时 | ✅ 开源 | ✅ 闭源 |
| 编译器（hipcc） | ✅ clang++ | ✅ amdclang++ |
| 调试器 | rocgdb | ROCm Debugger for Windows |
| 性能分析 | ROCProfiler | Radeon GPU Profiler |
| HIPIFY 移植工具 | ✅ | ✅ |
| **数学库（rocBLAS 等）** | ✅ | ✅ |
| 基础算子库 | ✅ | ✅ |
| HIPBlasLT | ✅ | ✅（6.4.2+，gfx1101）|
| 通信库（RCCL） | ✅ | ❌ 不可用 |
| **AI 库（MIOpen、MIGraphX）** | ✅ | ❌ **不可用** |
| 系统管理（ROCm SMI） | ✅ 完整 | 仅 hipInfo |
| AI 框架（PyTorch、TF 等） | ✅ | ❌ 不可用 |
| CMake HIP 语言支持 | ✅ | ❌ 不支持 |
| Visual Studio 插件 | N/A | ✅ 有插件 |
| HIP Ray Tracing | ✅ | ✅ |

---

## 4. 对 CCCC 项目的影响

`cccc-hip` 模块依赖：
- **rocBLAS**：矩阵乘法 → Windows/Linux 均可用
- **MIOpen**：卷积等算子 → **仅 Linux 可用**

| 平台 | rocBLAS | MIOpen | 状态 |
|------|---------|--------|------|
| Linux | ✅ | ✅ | 完整功能 |
| Windows | ✅ | ❌ | 卷积需 fallback（im2col+GEMM）|

**Windows 上的替代方案**：
1. 卷积使用自实现的 im2col + rocBLAS GEMM（已有，但速度慢）
2. 若需 Windows 深度学习推理，考虑扩展 **Vulkan Compute** 后端
3. 部署场景可通过 **ONNX Runtime + DirectML** 路径

---

## 5. MIOpen vs im2col+GEMM 性能对比（实测）

测试条件：N=4，输入 28×28，卷积 64→128 通道，3×3 kernel

| 方式 | 耗时 | 备注 |
|------|------|------|
| 自实现 im2col + rocBLAS GEMM | ~0.287 ms | 含 im2col 内存展开开销 |
| MIOpen 卷积 | ~0.010 ms | 使用 Winograd/implicit GEMM 等优化 kernel |
| **加速比** | **~28×** | MIOpen 显著优于自实现 |

MIOpen 在小 batch 场景下使用 Winograd 变换，大幅减少乘法次数；自实现方案在小 batch 时内存开销占比较高。

---

## 6. MIOpen Windows 编译方案

### 6.1 官方支持现状

**结论：官方 HIP SDK 不含 MIOpen，但可从源码编译获得 MIOpen.dll（ROCm 6.2+ 已有人成功）。**

| 状态 | 说明 |
|------|------|
| 官方 HIP SDK（Windows） | **不包含** MIOpen，需自行编译 |
| MIOpen 代码库 | 已迁移至 [ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries)（原仓库标注 retired）|
| Windows 构建 CI | 仅测试 MI 系列（数据中心卡），gfx110x 消费卡未纳入 CI |
| gfx1100（RX 7900 XTX）Windows | **可以编译并正常运行**（实测可用，Mar 2025）|
| gfx1200（RX 9xxx）Windows | 截至 ROCm 7.2 仍有 `HIPRTC_ERROR_COMPILATION` 错误（issue #3862）|

> AMD 官方工程师建议使用 WSL，但实际 Windows 原生构建已可用于 gfx1100。

### 6.2 从源码编译 MIOpen.dll（ROCm 7.1，实测可用方案）

> **适用版本**：MIOpen 3.5.2（仓库已迁移至 `ROCm/rocm-libraries`，源码位于 `projects/miopen`）
>
> ROCm 6.2 旧方案（需 Boost、lost-files 补丁）已过时，不再适用。

**前提条件**

| 工具 | 说明 |
|------|------|
| Visual Studio 2022 | 提供 MSVC 工具链和 CMake |
| ROCm HIP SDK 7.1 | 含 clang++ 21、amd_comgr、hiprtc、rocblas |
| CMake ≥ 3.15 | VS2022 自带即可 |
| Ninja | 如 `C:\project\bin\ninja.exe` |
| bzip2.exe | 用于解压 kernel DB 文件 |
| rocm-cmake | CMake 模块，如 `C:\build-tools\rocm-cmake` |
| BZip2 静态库 | 如编译安装到 `C:\bzip2-install` |
| nlohmann_json | header-only，如 `C:\nlohmann-json` |
| Half | header-only，如 `C:\half`（含 `half/half.hpp`）|
| vcpkg | 已有即可，如 `C:\project\microsoft\vcpkg` |

> **不再需要**：Boost、miopen-lost-files.zip、unistd.h 补丁、修改 CMakeLists.txt

**步骤一：用 vcpkg 安装 AI tuning 依赖**

```powershell
# 以下三个包均为 header-only，安装很快
.\vcpkg install frugally-deep:x64-windows eigen3:x64-windows
# functionalplus 会作为 frugally-deep 的依赖自动安装
```

**步骤二：克隆源码**

```powershell
git clone https://github.com/ROCm/rocm-libraries.git --depth=1 --branch develop
cd rocm-libraries\projects\miopen
```

**步骤三：打两处本地补丁（upstream 尚未修复）**

**补丁 1** — `src/include/miopen/hipoc_kernel.hpp`，在 `#include <cstring>` 后添加一行：
```cpp
#include <functional>
```
原因：`std::function` 被使用但未 include，clang++ 21 报错。

**补丁 2** — `src/include/miopen/fusion/problem_description.hpp`，在 `FusionDescription` 结构体的 `#if !MIOPEN_ENABLE_SQLITE` 块内（`Serialize` 函数之后）添加：
```cpp
template <class Self, class F>
static void Visit(Self&& self, F f)
{
    auto conv_prob = self.GetConvProblem(conv::Direction::Forward);
    conv::ProblemDescription::Visit(conv_prob, f);
}

template <class Self, class F>
static void VisitAll(Self&& self, F f)
{
    auto conv_prob = self.GetConvProblem(conv::Direction::Forward);
    conv::ProblemDescription::VisitAll(conv_prob, f);
}
```
原因：`FindRecord` 模板实例化要求 problem 类型提供 `Visit`/`VisitAll`，`FusionDescription` 缺少。

**步骤四：创建 amd_comgr CMake 配置文件（ROCm 7.1 遗漏）**

ROCm 7.1 Windows 安装包**遗漏**了 `amd_comgr` 的 CMake 包配置文件，导致 CMake 报错 "Could not find amd_comgr-config.cmake"。需手动创建 4 个文件到本地目录（无需管理员权限）：

```powershell
$dir = "C:\build-tools\amd_comgr-cmake"
New-Item -ItemType Directory -Force $dir | Out-Null

@'
get_filename_component(AMD_COMGR_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
include("${AMD_COMGR_PREFIX}/amd_comgr-targets.cmake")
'@ | Set-Content "$dir\amd_comgr-config.cmake"

@'
set(PACKAGE_VERSION "3.0.0")
set(PACKAGE_VERSION_EXACT FALSE)
set(PACKAGE_VERSION_COMPATIBLE TRUE)
if("${PACKAGE_FIND_VERSION}" VERSION_GREATER "3.0.0")
  set(PACKAGE_VERSION_COMPATIBLE FALSE)
endif()
'@ | Set-Content "$dir\amd_comgr-config-version.cmake"

@'
add_library(amd_comgr SHARED IMPORTED)
set_target_properties(amd_comgr PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "C:/Program Files/AMD/ROCm/7.1/include"
)
get_filename_component(_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
include("${_IMPORT_PREFIX}/amd_comgr-targets-release.cmake")
'@ | Set-Content "$dir\amd_comgr-targets.cmake"

@'
set_target_properties(amd_comgr PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_IMPLIB_RELEASE "C:/Program Files/AMD/ROCm/7.1/lib/amd_comgr0701.lib"
  IMPORTED_LOCATION_RELEASE "C:/Program Files/AMD/ROCm/7.1/bin/amd_comgr0701.dll"
)
'@ | Set-Content "$dir\amd_comgr-targets-release.cmake"
```

> **说明**：DLL 和头文件本身存在（`bin\amd_comgr0701.dll`、`lib\amd_comgr0701.lib`、`include\amd_comgr\`），仅 CMake 集成层缺失。ROCm 6.x 安装包含这些文件（`lib\cmake\amd_comgr\`），ROCm 7.1 打包遗漏。

**步骤五：CMake 配置**

```powershell
$cmake = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$env:PATH = "C:\Program Files\AMD\ROCm\7.1\bin;" + $env:PATH
$prefix = "C:/build-tools/rocm-cmake;C:/Program Files/AMD/ROCm/7.1;C:/Program Files/AMD/ROCm/7.1/hip;C:/project/microsoft/vcpkg/installed/x64-windows"

& $cmake -G Ninja -S . -B build `
  -DCMAKE_CXX_COMPILER="C:/Program Files/AMD/ROCm/7.1/bin/clang++.exe" `
  -DCMAKE_C_COMPILER="C:/Program Files/AMD/ROCm/7.1/bin/clang.exe" `
  -DCMAKE_MAKE_PROGRAM="C:/project/bin/ninja.exe" `
  "-DCMAKE_PREFIX_PATH=$prefix" `
  -Damd_comgr_DIR="C:/build-tools/amd_comgr-cmake" `
  -DMIOPEN_BACKEND=HIP -DBUILD_SHARED_LIBS=ON `
  -DMIOPEN_USE_COMPOSABLEKERNEL=OFF -DMIOPEN_USE_MLIR=OFF `
  -DMIOPEN_ENABLE_SQLITE=OFF -DMIOPEN_ENABLE_SQLITE_KERN_CACHE=OFF `
  -DMIOPEN_USE_SQLITE_PERFDB=ON -DBUILD_TESTING=OFF `
  -DMIOPEN_BUILD_DRIVER=OFF `
  -DMIOPEN_ENABLE_AI_KERNEL_TUNING=ON `
  -DMIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK=ON `
  -DBZIP2_LIBRARIES="C:/bzip2-install/lib/bz2_static.lib" `
  -DBZIP2_INCLUDE_DIR="C:/bzip2-install/include" `
  -DBZIP2_LIBRARY_RELEASE="C:/bzip2-install/lib/bz2_static.lib" `
  -Dnlohmann_json_DIR="C:/nlohmann-json" `
  -DHALF_INCLUDE_DIR="C:/half" `
  -DUNZIPPER="C:/project/bin/bzip2.exe"
```

**步骤六：编译**

```powershell
# 需要将 Windows SDK 的 mt.exe 加入 PATH（新版本新增 addkernels.manifest 需要）
$env:PATH = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64;" + $env:PATH
Set-Location build
C:\project\bin\ninja.exe -j8
```

编译完成后 `build\bin\MIOpen.dll` 约 490 MB。

### 6.3 Windows vs Linux 性能对比（MIOpen 实测）

以下为 RX 7900 XTX 实测数据（时间单位：秒，越小越好）：

| 场景 | Windows | Linux (Fedora) | 胜出 |
|------|---------|----------------|------|
| CNN 训练（分类） | **290** | 338 | Windows |
| Tune（算法搜索） | 236 | **150** | Linux |
| 全连接层 | **127** | 133 | Windows |

**结论**：Windows 上的 MIOpen 在部分场景下不比 Linux 差，甚至更快。Linux 的 Tune 阶段更快可能源于更好的 kernel 缓存机制。

### 6.4 已知问题（截至 MIOpen 3.5.2 / ROCm 7.1）

| 问题 | 根因 | 解决方法 |
|------|------|---------|
| `hipoc_kernel.hpp` 缺少 `#include <functional>` | upstream 漏提交 | 手动添加（本地补丁 1）|
| `FusionDescription` 缺少 `Visit`/`VisitAll` | upstream 未实现 | 手动添加（本地补丁 2）|
| 3.5.2 upstream bug：AI tuning include guard 错误 | `HeuristicInitState` 在 guard 内但被 guard 外代码引用 | 启用 `MIOPEN_ENABLE_AI_KERNEL_TUNING=ON`（不再 guard，头文件完整可见）|
| 编译时找不到 `mt.exe` | 新增 `addkernels/addkernels.manifest` 触发清单嵌入 | 将 Windows SDK `bin\x64\` 加入 PATH |
| gfx1200 `HIPRTC_ERROR_COMPILATION` | ROCm 7.x 未解决 | 临时绕过：`MIOPEN_DEBUG_CONV_DIRECT=0` |
| `ComposableKernel` 和 `rocMLIR` 无法在 Windows 编译 | 依赖 Linux 工具链 | 编译时关闭，对 gfx1100 日常性能影响不大 |
| CMake 报错 "Could not find amd_comgr-config.cmake" | ROCm 7.1 安装包遗漏 cmake 包配置文件（ROCm 6.x 有，7.1 无）| 手动创建 4 个 cmake 文件到 `C:\build-tools\amd_comgr-cmake\`，cmake 加 `-Damd_comgr_DIR`（见步骤四）|

### 6.5 参考链接

- [How to build MIOpen.dll on Windows（Discussion #2703）](https://github.com/ROCm/MIOpen/discussions/2703)
- [ROCm/rocm-libraries（MIOpen 新仓库）](https://github.com/ROCm/rocm-libraries/tree/develop/projects/miopen)
- [gfx1200 Windows MIOpen issue #3862](https://github.com/ROCm/MIOpen/issues/3862)
- [WSL2 安装 ROCm 官方指南](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-radeon.html)
