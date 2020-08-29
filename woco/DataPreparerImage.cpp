#include "DataPreparerImage.h"
#include "File.h"
#include "Option.h"
#include "Timer.h"
#include "convert.h"
#include <atomic>
#include <ctime>
#include <iostream>

#define _USE_MATH_DEFINES
#include <cmath>

namespace woco
{

DataPreparerImage::DataPreparerImage()
{
}

void DataPreparerImage::init2()
{
    OPTION_GET_INT(flip_);
    OPTION_GET_INT(transpose_);

    OPTION_GET_INT(d_channel_);
    OPTION_GET_REAL(d_noise_);

    OPTION_GET_NUMVECTOR(d_contrast_, "0,0", 2);
    OPTION_GET_NUMVECTOR(d_brightness_, "0,0", 2);

    //LOG("Options for image processing %s end\n\n", section_.c_str());
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
        if (floor(rand_.rand() * 2) - 1 != 0)
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

    if (d_brightness_[1] >= d_brightness_[0] && d_brightness_[0] != 0
        || d_contrast_[1] >= d_contrast_[0] && d_contrast_[0] != 0)
    {
        need_limit = true;
        auto b = d_brightness_[0] + (d_brightness_[1] - d_brightness_[0]) * rand_.rand();    //亮度
        auto c = 1 + d_contrast_[0] + (d_contrast_[1] - d_contrast_[0]) * rand_.rand();      //对比度
        for (int ic = 0; ic < X1.channel(); ic++)
        {
            if (d_channel_)
            {
                b = d_brightness_[0] + (d_brightness_[1] - d_brightness_[0]) * rand_.rand();
                c = 1 + d_contrast_[0] + (d_contrast_[1] - d_contrast_[0]) * rand_.rand();
            }
            for (int ih = 0; ih < X1.height(); ih++)
            {
                for (int iw = 0; iw < X1.width(); iw++)
                {
                    auto& v = X1.getData(iw, ih, ic, 0);
                    v = v * c + b;
                }
            }
        }
    }

    //printf("%f ", temp->getData(64, 64, 0, 0));
    if (need_limit)
    {
        X1.sectionLimit(0, 1);
    }
    //printf("%f\n", temp->getData(64, 64, 0, 0));
}

}    // namespace woco