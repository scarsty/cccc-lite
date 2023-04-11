#include "Cifar10Reader.h"
#include "filefunc.h"
#include "Log.h"

namespace cccc
{

Cifar10Reader::Cifar10Reader()
{
}

static void readData(Matrix& X, Matrix& Y, const std::string& filename, int begin)
{
    std::vector<uint8_t> arr;
    filefunc::readFileToVector(filename, arr);
    int count = begin - 1;
    int k = 0;
    for (int i = 0; i < arr.size(); i++)
    {
        if (i % 3073 == 0)
        {
            count++;
            Y.getData(arr[i], count) = 1;
            k = 0;
        }
        else
        {
            X.getData(k++, count) = arr[i] / 255.0;
        }
    }
    //for (int in = begin; in < begin+10000; in++)
    //{
    //    for (int ic = 0; ic < 3; ic++)
    //    {
    //        real min1 = 1, max1 = 0;
    //        for (int iw = 0; iw < 32; iw++)
    //        {
    //            for (int ih = 0; ih < 32; ih++)
    //            {
    //                auto v = X.getData(iw, ih, ic, in);
    //                min1 = std::min(v,min1);
    //                max1 = std::max(v, max1);
    //            }
    //        }
    //        for (int iw = 0; iw < 32; iw++)
    //        {
    //            for (int ih = 0; ih < 32; ih++)
    //            {
    //                auto& v = X.getData(iw, ih, ic, in);
    //                v /= max1 - min1;
    //            }
    //        }
    //    }
    //}
}

void Cifar10Reader::load(Matrix& X, Matrix& Y, std::string path /*= "cifa10"*/, int data_type /*= 1*/)
{
    if (path.back() != '/' || path.back() != '\\')
    {
        path += '/';
    }
    if (data_type == 1)
    {
        X.resize(32, 32, 3, 50000);
        Y.resize(10, 50000);
        Y.initData(0);
        LOG("Loading Cifar10 train data... ");
        readData(X, Y, path + "data_batch_1.bin", 0);
        readData(X, Y, path + "data_batch_2.bin", 10000);
        readData(X, Y, path + "data_batch_3.bin", 20000);
        readData(X, Y, path + "data_batch_4.bin", 30000);
        readData(X, Y, path + "data_batch_5.bin", 40000);
        LOG("done\n");
    }
    else if (data_type == 2)
    {
        X.resize(32, 32, 3, 10000);
        Y.resize(10, 10000);
        Y.initData(0);
        LOG("Loading Cifar10 test data... ");
        readData(X, Y, path + "test_batch.bin", 0);
        LOG("done\n");
    }
    else
    {
        LOG("Please check Cifar10 type: 1 - train set (50000), 2 - test set (10000)\n");
        return;
    }
    //X.message();
    //Y.message();
}

}    // namespace cccc
