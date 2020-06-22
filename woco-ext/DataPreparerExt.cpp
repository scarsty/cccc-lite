#include "DataPreparerExt.h"
#include "Cifar10Reader.h"
#include "ConsoleControl.h"
#include "MnistReader.h"
#include "Option.h"

#define _USE_MATH_DEFINES
#include <cmath>

namespace woco
{

DataPreparerExt::DataPreparerExt()
{
}

void DataPreparerExt::init2()
{
    DataPreparerImage::init2();
    OPTION_GET_INT(type_);
    OPTION_GET_INT(random_diff_);
    OPTION_GET_INT(remove59_);
    OPTION_GET_INT(aem_);
}

//one example to deal MNIST
void DataPreparerExt::fillData(Matrix& X, Matrix& Y)
{
    if (fill_times_ == 0)
    {
        std::string format = Option::getInstance().getString(section_, "format", "mnist");
        if (format == "mnist")
        {
            //使用MNIST库，通常用来测试网络
            MnistReader loader;
            std::string path = Option::getInstance().getString(section_, "path", "mnist");
            loader.load(X, Y, path, type_);
        }
        else if (format == "cifar10")
        {
            Cifar10Reader loader;
            std::string path = Option::getInstance().getString(section_, "path", "cifar10");
            loader.load(X, Y, path, type_);
        }

        //data.save(mnist_path+"/train.bin");
        if (remove59_)
        {
            //如果需要进行旋转和转置的实验，则5和9不适合留在数据集
            int count = 0;
            for (int i = 0; i < X.getNumber(); i++)
            {
                if (Y.getData(0, 0, 5, i) == 1 || Y.getData(0, 0, 9, i) == 1)
                {
                    count++;
                }
            }
            int k = 0;
            for (int i = 0; i < X.getNumber(); i++)
            {
                if (Y.getData(0, 0, 5, i) != 1 && Y.getData(0, 0, 9, i) != 1)
                {
                    Matrix::copyDataPointer(X, X.getDataPointer(0, 0, 0, i), X, X.getDataPointer(0, 0, 0, k), X.getRow());
                    Matrix::copyDataPointer(Y, Y.getDataPointer(0, 0, 0, i), Y, Y.getDataPointer(0, 0, 0, k), Y.getRow());
                    k++;
                }
            }
            X.resizeNumber(X.getNumber() - count);
            Y.resizeNumber(Y.getNumber() - count);
        }

        if (aem_)
        {
            Y = X;
        }
    }
    fill_times_++;
}

void DataPreparerExt::transOne(Matrix& X1, Matrix& Y1)
{
    DataPreparerImage::transOne(X1, Y1);
    if (random_diff_ && !X1.inGPU())
    {
        int diff_x_ = rand_.rand() * 9 - 4, diff_y_ = rand_.rand() * 9 - 4;
        Matrix x1c = X1.clone(DeviceType::CPU);
        X1.initData(0);
        for (int ix = 0; ix < 20; ix++)
        {
            for (int iy = 0; iy < 20; iy++)
            {
                X1.getData(4 + diff_x_ + ix, 4 + diff_y_ + iy, 0, 0) = x1c.getData(4 + ix, 4 + iy, 0, 0);
            }
        }
    }
}

/*
    注意，以下部分包含静态变量：

    Log
    ConsoleControl
    DynamicLibrary
    CudaControl
    Option

    如j静态链接核心库，则需注意dll和exe中的静态变量不同    

    此处需特别注意clone的操作，clone的默认操作是优先复制到GPU，但是核心库静态链接时，CudaControl的全局变量在插件dll中包含一个副本
    此时clone的默认操作是复制到CPU，若改为动态链接则是复制到GPU，故此处应明确写出参数
    
    在不同操作系统中，全局变量的链接行为也可能存在区别，请参考相关的文档
*/
}    // namespace woco