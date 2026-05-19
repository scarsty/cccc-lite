#pragma once
#include <string>

#ifdef _WIN32
#ifdef CCCC_SD_BUILDING
#define CCCC_SD_API __declspec(dllexport)
#else
#define CCCC_SD_API __declspec(dllimport)
#endif
#else
#define CCCC_SD_API
#endif

typedef void* SdHandle;

// 每个 DDIM 采样步完成后回调，用于显示进度。
// step:       当前步编号（从 1 开始）
// total:      总步数
// userdata:   透传指针
typedef void (*SdProgressCallback)(int step, int total, void* userdata);

// 初始化 Stable Diffusion Pipeline。
// model_dir:  包含所有 ncnn/cccc 模型文件的目录。
// 返回句柄，失败返回 nullptr。
CCCC_SD_API SdHandle sd_init(const std::string& model_dir);

// 释放所有资源。
CCCC_SD_API void sd_destroy(SdHandle handle);

// 文生图主接口（DDIM 采样）。
// prompt:         正向文本提示
// output_path:    输出图片路径（png）
// steps:          DDIM 采样步数（默认 20）
// guidance_scale: 无分类引导系数（默认 7.5）
// width, height:  输出分辨率（应为 8 的倍数，默认 512×512）
// seed:           随机种子（-1 表示随机）
// callback:       可选进度回调
// 返回 0 表示成功，非 0 表示失败。
//
CCCC_SD_API int sd_generate(SdHandle handle,
    const std::string& prompt,
    const std::string& output_path,
    int steps,
    float guidance_scale,
    int width,
    int height,
    int seed,
    SdProgressCallback callback,
    void* userdata);
