#include "DataPreparerImage.h"
#include "filefunc.h"
#include "Timer.h"
#include "strfunc.h"
#include <atomic>
#include <ctime>
#include <iostream>

#define _USE_MATH_DEFINES
#include <cmath>

namespace cccc
{

DataPreparerImage::DataPreparerImage()
{
}

DataPreparerImage::~DataPreparerImage()
{
}

void DataPreparerImage::init2()
{
    OPTION_GET_INT(flip_);
    OPTION_GET_INT(transpose_);

    OPTION_GET_INT(d_channel_);
    OPTION_GET_REAL(d_noise_);

    OPTION_GET_NUMVECTOR(d_contrast_, "0,0", 2, 0);
    OPTION_GET_NUMVECTOR(d_brightness_, "0,0", 2, 0);

    //LOG("Options for image processing {} end\n\n", section_);
}

//变换一张图
//此处将原始数据进行一些干扰，适应不同的情况
void DataPreparerImage::transOne(Matrix& X1, Matrix& Y1)
{
    bool need_limit = false;    //图片的值需要限制到0~1

    if (flip_ != 0)
    {
        //flip -1, 0, 1, other value means do nothing
        int f = floor(rand_.rand() * 4) - 1;
        X1.flip(f);
    }
    if (transpose_ != 0)
    {
        //transpose 0, 1
        int t = floor(rand_.rand() * 2) - 1;
        if (t)
        {
            X1.transpose();
        }
    }

    //噪点
    if (d_noise_ != 0 && X1.getDeviceType() == DeviceType::CPU)
    {
        need_limit = true;
        for (int i = 0; i < X1.getDataSize(); i++)
        {
            X1.getData(i, 0) += rand_.rand() * d_noise_ * 2 - d_noise_;
        }
    }

    //亮度和对比度连y一起变换
    //亮度
    if (d_brightness_[1] >= d_brightness_[0] && d_brightness_[0] != 0)
    {
        need_limit = true;
        if (d_channel_ == 0)
        {
            auto b = d_brightness_[0] + (d_brightness_[1] - d_brightness_[0]) * rand_.rand();
            X1.addNumber(b);
        }
        else
        {
            for (int i = 0; i < X1.getNumber(); i++)
            {
                auto b = d_brightness_[0] + (d_brightness_[1] - d_brightness_[0]) * rand_.rand();
                X1.addNumberCol(b, 1, i);
            }
        }
    }
    //对比度
    if (d_contrast_[1] >= d_contrast_[0] && d_contrast_[0] != 0)
    {
        need_limit = true;
        if (d_channel_ == 0)
        {
            auto c = 1 + d_contrast_[0] + (d_contrast_[1] - d_contrast_[0]) * rand_.rand();
            X1.scale(c);
        }
        else
        {
            for (int i = 0; i < X1.getNumber(); i++)
            {
                auto c = 1 + d_contrast_[0] + (d_contrast_[1] - d_contrast_[0]) * rand_.rand();
                X1.scaleCol(c, i);
            }
        }
    }
    if (need_limit)
    {
        X1.sectionLimit(0, 1);
    }
}

}    // namespace cccc