#include "DataPreparerMnist.h"
#include "Cifar10Reader.h"
#include "ConsoleControl.h"
#include "MnistReader.h"

#define _USE_MATH_DEFINES
#include <cmath>

namespace cccc
{

DataPreparerMnist::DataPreparerMnist()
{
}

DataPreparerMnist::~DataPreparerMnist()
{
    //LOG("Destory MNIST with section [{}]\n", section_);
}

//one example to deal MNIST
void DataPreparerMnist::initData()
{
    std::string format = option_->getString(section_, "format", "mnist");
    if (format == "mnist")
    {
        //使用MNIST库，通常用来测试网络
        MnistReader loader;
        std::string path = option_->getString(section_, "path", "mnist");
        loader.load(X0, Y0, path, type_);
    }
    else if (format == "cifar10")
    {
        Cifar10Reader loader;
        std::string path = option_->getString(section_, "path", "cifar10");
        loader.load(X0, Y0, path, type_);
    }

    //data.save(mnist_path+"/train.bin");
    if (remove59_)
    {
        //如果需要进行旋转和转置的实验，则5和9不适合留在数据集
        int count = 0;
        for (int i = 0; i < X0.getNumber(); i++)
        {
            if (Y0.getData(0, 0, 5, i) == 1 || Y0.getData(0, 0, 9, i) == 1)
            {
                count++;
            }
        }
        auto X1 = X0.clone();
        auto Y1 = Y0.clone();
        auto new_number = X0.getNumber() - count;
        X0.resizeNumber(new_number);
        Y0.resizeNumber(new_number);
        int k = 0;
        for (int i = 0; i < X1.getNumber(); i++)
        {
            if (Y1.getData(0, 0, 5, i) != 1 && Y1.getData(0, 0, 9, i) != 1)
            {
                Matrix::copyRows(X1, i, X, k, 1);
                Matrix::copyRows(Y1, i, Y, k, 1);
                k++;
            }
        }
    }

    if (reverse_color_)
    {
        for (int i = 0; i < X0.getDataSize(); i++)
        {
            X0.getData(i) = 1 - X0.getData(i);
        }
    }
    LOG("Recheck fill_group: ");
    fill_group_ = X0.getNumber();
    OPTION_GET_INT(fill_group_);
    DataPreparer::initData();
}

void DataPreparerMnist::init2()
{
    DataPreparerImage::init2();
    if (section_.find("test") != std::string::npos)
    {
        type_ = 2;
    }
    OPTION_GET_INT(type_);
    OPTION_GET_INT(random_diff_);
    OPTION_GET_INT(remove59_);
    OPTION_GET_INT(reverse_color_);
}

void DataPreparerMnist::fillData0()
{
}

void DataPreparerMnist::transOne(Matrix& X1, Matrix& Y1)
{
    if (random_diff_ && X1.getDeviceType() == UnitType::CPU)
    {
        int diff_x_ = rand_.rand() * 9 - 4, diff_y_ = rand_.rand() * 9 - 4;
        Matrix x1c = X1.clone(UnitType::CPU);
        X1.fillData(0);
        for (int ix = 0; ix < 20; ix++)
        {
            for (int iy = 0; iy < 20; iy++)
            {
                X1.getData(4 + diff_x_ + ix, 4 + diff_y_ + iy, 0, 0) = x1c.getData(4 + ix, 4 + iy, 0, 0);
            }
        }
    }
    DataPreparerImage::transOne(X1, Y1);
}

/*
    注意，以下部分包含静态变量：

    Log
    ConsoleControl
    DynamicLibrary
    CudaDeviceControl

    如在外置dll中使用，则需注意静态变量与exe中不同

    由于CudaDeviceControl以静态量记录设备信息，故在外置dll中不可以进行矩阵的GPU相关操作，如有此需求请考虑动态链接

    此处需特别注意clone的操作，clone的默认操作是优先复制到GPU，但是核心库静态链接时，CudaDevice的全局变量在插件dll中包含一个副本
    此时clone的默认操作是复制到CPU，若改为动态链接则是复制到GPU，故此处应明确写出参数

    在不同操作系统中，全局变量的链接行为也可能存在区别，请参考相关的文档
*/
}    // namespace cccc