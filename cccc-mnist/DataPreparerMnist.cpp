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
        //ʹ��MNIST�⣬ͨ��������������
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
        //�����Ҫ������ת��ת�õ�ʵ�飬��5��9���ʺ��������ݼ�
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
            X0.setData(i, 1 - X0.getData(i));
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
                X1.setData(4 + diff_x_ + ix, 4 + diff_y_ + iy, 0, 0, x1c.getData(4 + ix, 4 + iy, 0, 0));
            }
        }
    }
    DataPreparerImage::transOne(X1, Y1);
}

/*
    ע�⣬���²��ְ�����̬������

    Log
    ConsoleControl
    DynamicLibrary
    CudaDeviceControl

    ��������dll��ʹ�ã�����ע�⾲̬������exe�в�ͬ

    ����CudaDeviceControl�Ծ�̬����¼�豸��Ϣ����������dll�в����Խ��о����GPU��ز��������д������뿼�Ƕ�̬����

    �˴����ر�ע��clone�Ĳ�����clone��Ĭ�ϲ��������ȸ��Ƶ�GPU�����Ǻ��Ŀ⾲̬����ʱ��CudaDevice��ȫ�ֱ����ڲ��dll�а���һ������
    ��ʱclone��Ĭ�ϲ����Ǹ��Ƶ�CPU������Ϊ��̬�������Ǹ��Ƶ�GPU���ʴ˴�Ӧ��ȷд������

    �ڲ�ͬ����ϵͳ�У�ȫ�ֱ�����������ΪҲ���ܴ���������ο���ص��ĵ�
*/
}    // namespace cccc