#!/usr/bin/env python3
"""
trim_lite.py — cccc-lite 剪裁脚本
================================================
将 lite 版不需要的反向函数体清空，仅保留函数签名和空 {}。

按照 lite版的区别.md 的要求：
  - 反向仅支持 Relu/Softmax/Sigmoid/Pool/Conv/全连接等基本算子。
  - MatrixEx 中其他反向实现缺失，仅保留函数名。
  - cccc-cuda 和 cccc-hip 中其他反向实现缺失，仅保留函数名。

可重复运行（幂等）。第一次运行时会自动备份原文件（*.trim_backup）。

用法：
    cd c:\\project\\cccc-lite
    python trim_lite.py
"""
import re
import sys
import shutil
import os


def replace_patterns_in_file(filepath, replacements, description):
    """
    使用正则替换文件内容；支持幂等运行。
    replacements: [(pattern, repl), ...]
    """
    if not os.path.exists(filepath):
        print(f"  文件不存在，跳过: {filepath}")
        return False

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    old_text = text
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.MULTILINE)

    if text == old_text:
        print(f"  无需修改: {filepath}")
        return False

    backup = filepath + '.trim_backup'
    if not os.path.exists(backup):
        shutil.copy2(filepath, backup)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"  已保存: {filepath} ({description})")
    return True

def clear_function_bodies(filepath, func_names, is_method=True, return_zero=False):
    """
    在 C++ 文件中找到指定函数并清空函数体（保留签名和空 {}）。
    func_names: 函数名列表（不含命名空间前缀）
    is_method: True=方法（有类名::前缀），False=普通函数
    return_zero: True=清空为 { return 0; }（C 风格返回 int）
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    changed = False
    for name in func_names:
        # 匹配函数签名：从函数名开始到第一个 {
        # 支持跨行的参数列表
        if is_method:
            pattern = r'(void\s+MatrixEx::' + re.escape(name) + r'\s*\([^)]*(?:\([^)]*\)[^)]*)*\)\s*(?:\{|(?:[^;{]*\{)))'
        else:
            # C 导出函数: int func_name(...)
            pattern = r'((?:int|void)\s+' + re.escape(name) + r'\s*\([^)]*\)\s*\{)'

        # 更健壮的搜索：找到函数名在文件中的所有位置
        # 先找 "FuncName(" 的所有位置
        search_pattern = name + r'\s*\('
        positions = [m.start() for m in re.finditer(search_pattern, text)]

        replaced_in_func = False
        for pos in positions:
            # 从该位置向前找函数签名开始（查找 void 或 int）
            # 向后找第一个 {
            brace_pos = text.find('{', pos)
            if brace_pos == -1:
                continue

            # 验证这是函数定义（在 { 之前没有 ; ）
            between = text[pos:brace_pos]
            if ';' in between:
                continue  # 这是声明，不是定义

            # 找对应的匹配 }
            depth = 0
            i = brace_pos
            end_pos = -1
            while i < len(text):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        end_pos = i
                        break
                i += 1

            if end_pos == -1:
                print(f"  警告: {name} 找不到匹配的 }}")
                continue

            # 当前函数体内容
            body = text[brace_pos+1:end_pos]

            # 如果已经是空体（只有空白），跳过
            if return_zero:
                replacement_body = '\n    return 0;\n'
            else:
                replacement_body = ''

            if body.strip() == replacement_body.strip():
                # print(f"  {name}: 已是空体，跳过")
                break

            # 替换函数体
            new_body = '{' + replacement_body + '}'
            text = text[:brace_pos] + new_body + text[end_pos+1:]
            print(f"  已清空: {name}")
            replaced_in_func = True
            changed = True
            break

        if not replaced_in_func and name not in [m.group() for m in re.finditer(name + r'\s*\(', text) if False]:
            # 简单检查函数是否存在
            if name + '(' not in text and name + ' (' not in text:
                print(f"  未找到: {name}")

    if changed:
        # 备份
        backup = filepath + '.trim_backup'
        if not os.path.exists(backup):
            shutil.copy2(filepath, backup)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"  已保存: {filepath}")
    else:
        print(f"  无需修改: {filepath}")

    return changed


def clear_c_functions(filepath, func_names, return_zero=False):
    """
    清空 C/CUDA/HIP 风格函数体。
    return_zero=True 时函数体替换为 { return 0; }，否则替换为 {}。
    """
    if not os.path.exists(filepath):
        print(f"  文件不存在，跳过: {filepath}")
        return False

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    changed = False
    for name in func_names:
        positions = [m.start() for m in re.finditer(re.escape(name) + r'\s*\(', text)]
        for pos in positions:
            brace_pos = text.find('{', pos)
            if brace_pos == -1:
                continue
            between = text[pos:brace_pos]
            if ';' in between:
                continue

            depth = 0
            i = brace_pos
            end_pos = -1
            while i < len(text):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        end_pos = i
                        break
                i += 1

            if end_pos == -1:
                continue

            body = text[brace_pos + 1:end_pos]
            replacement_body = '\n    return 0;\n' if return_zero else ''
            if body.strip() == replacement_body.strip():
                break

            text = text[:brace_pos] + '{' + replacement_body + '}' + text[end_pos + 1:]
            print(f"  已清空: {name}")
            changed = True
            break

    if changed:
        backup = filepath + '.trim_backup'
        if not os.path.exists(backup):
            shutil.copy2(filepath, backup)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"  已保存: {filepath}")
    else:
        print(f"  无需修改: {filepath}")

    return changed


def main():
    base = os.path.dirname(os.path.abspath(__file__))

    print("=== 剪裁 MatrixEx.cpp ===")
    matrixex_file = os.path.join(base, 'cccc', 'MatrixEx.cpp')

    # 需要清空的非基本算子反向函数
    matrixex_funcs_to_clear = [
        'concatByChannelBackward',
        'chunkBackward',
        'dropoutBackward',
        'batchNormalizationBackward',
        'layerNormalizationBackward',
        'correlationBackward',
        'rmsNormBackward',
        'permute4dBackward',
        'ropeBackward',
        'pixelShuffleBackward',
        'attentionBackward',
        'embedBackward',
        'tileBackward',
        'deconvolutionBackwardDA',
        'deconvolutionBackwardDW',
        'groupNormBackward',
        'reparamBackward',
        'l1LossBackward',
        'klLvBackward',
        'upsampleBackward',
    ]

    clear_function_bodies(matrixex_file, matrixex_funcs_to_clear, is_method=True, return_zero=False)

    print("\n=== 剪裁 cuda_functions.cu ===")
    cuda_file = os.path.join(base, 'cccc-cuda', 'cuda_functions.cu')

    # cuda 中的 C 导出反向函数（返回 int），清空为 return 0;
    cuda_bwd_funcs = [
        'cuda_layer_norm_bwd',
        'cuda_rms_norm_bwd',
        'cuda_rope_bwd',
        'cuda_pixel_shuffle_bwd',
        'cuda_embed_bwd',
        'cuda_tile_bwd',
        'cuda_group_norm_affine_bwd',
        'cuda_reparam_bwd',
        'cuda_l1_bwd',
        'cuda_kl_lv_bwd',
        'cuda_upsample_nearest_bwd',
        'cuda_upsample_bilinear_bwd',
    ]

    cuda_bwd_kernels = [
        'layer_norm_bwd_float_kernel',
        'rms_norm_bwd_float_kernel',
        'rope_bwd_float_kernel',
        'pixel_shuffle_bwd_kernel',
        'embed_bwd_float_kernel',
        'tile_bwd_float_kernel',
        'group_norm_affine_bwd_float',
        'reparam_bwd_float',
        'l1_bwd_float',
        'kl_lv_bwd_float',
        'upsample_nearest_bwd_float',
        'upsample_bilinear_bwd_float',
    ]
    # 入口函数清空为 return 0;，对应核函数清空为 {}
    clear_c_functions(cuda_file, cuda_bwd_funcs, return_zero=True)
    clear_c_functions(cuda_file, cuda_bwd_kernels, return_zero=False)

    print("\n=== 剪裁 hip_functions.cpp ===")
    hip_file = os.path.join(base, 'cccc-hip', 'hip_functions.cpp')

    # hip 中的 C 导出反向函数（返回 int），清空为 return 0;
    hip_bwd_funcs = [
        'hip_layer_norm_bwd',
        'hip_rms_norm_bwd',
        'hip_rope_bwd',
        'hip_pixel_shuffle_bwd',
        'hip_embed_bwd',
        'hip_tile_bwd',
        'hip_group_norm_affine_bwd',
        'hip_reparam_bwd',
        'hip_l1_bwd',
        'hip_kl_lv_bwd',
        'hip_upsample_nearest_bwd',
        'hip_upsample_bilinear_bwd',
    ]

    hip_bwd_kernels = [
        'layer_norm_bwd_float_kernel',
        'layer_norm_bwd_bf16_kernel',
        'rms_norm_bwd_float_kernel',
        'rms_norm_bwd_bf16_kernel',
        'rope_bwd_float_kernel',
        'rope_bwd_bf16_kernel',
        'pixel_shuffle_bwd_kernel',
        'pixel_shuffle16_bwd_kernel',
        'embed_bwd_float_kernel',
        'embed_bwd_bf16_kernel',
        'tile_bwd_float_kernel',
        'tile_bwd_bf16_kernel',
        'group_norm_affine_bwd_float',
        'group_norm_affine_bwd_bf16',
        'reparam_bwd_float',
        'reparam_bwd_bf16',
        'l1_bwd_float',
        'l1_bwd_bf16',
        'kl_lv_bwd_float',
        'kl_lv_bwd_bf16',
        'upsample_nearest_bwd_float',
        'upsample_nearest_bwd_bf16',
        'upsample_bilinear_bwd_float',
        'upsample_bilinear_bwd_bf16',
    ]

    # 入口函数清空为 return 0;，对应核函数清空为 {}
    clear_c_functions(hip_file, hip_bwd_funcs, return_zero=True)
    clear_c_functions(hip_file, hip_bwd_kernels, return_zero=False)

    print("\n=== 剪裁 cccc.vcxproj ===")
    vcxproj_file = os.path.join(base, 'cccc', 'cccc.vcxproj')
    replace_patterns_in_file(
        vcxproj_file,
        [
            (r'^\s*<Command>auto_version\.exe --version \$\(ProjectName\)\.rc --date2</Command>\s*\r?\n', ''),
            (r'^\s*<ClCompile Include="predict\.cpp"\s*/>\s*\r?\n', ''),
            (r'^\s*<ClInclude Include="predict\.h"\s*/>\s*\r?\n', ''),
            (r'^\s*<ResourceCompile Include="cccc\.rc"\s*/>\s*\r?\n', ''),
        ],
        'remove predict/cccc.rc/auto_version'
    )

    print("\n=== 强制单卡训练 ===")
    mainprocess_file = os.path.join(base, 'cccc', 'MainProcess.cpp')
    replace_patterns_in_file(
        mainprocess_file,
        [
            (
                r'MP_count_\s*=\s*option_\.getInt\("train",\s*"mp",\s*1\);\s*\r?\n\s*if\s*\(MP_count_\s*<=\s*0\)\s*\r?\n\s*\{\s*\r?\n\s*MP_count_\s*=\s*1;\s*\r?\n\s*\}',
                'MP_count_ = 1;'
            ),
        ],
        'force MP_count_ = 1'
    )

    print("\n=== 剪裁完成 ===")


if __name__ == '__main__':
    main()
