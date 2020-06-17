#include "DataPreparer.h"
#include "File.h"
#include "Option.h"
#include <ctime>

namespace woco
{

DataPreparer::DataPreparer()
{
}

void DataPreparer::init()
{
    if (Option::getInstance().hasSection(section_))
    {
        Log::setLog(Option::getInstance().getInt(section_, "output_log", 1));
    }
    else
    {
        Log::setLog(0);
    }
    OPTION_GET_INT(shuffle_);
    OPTION_GET_INT(trans_);
    if (create_by_dll_ != "")
    {
        fill_ = 1;
    }
    OPTION_GET_INT(fill_);
    OPTION_GET_INT(fill_group_);
    init2();
    Log::setLog(1);
}

//初始化训练集准备器
int DataPreparer::getFillGroup()
{
    if (fill_)
    {
        return fill_group_;
    }
    return -1;
}

//变换一组数据，并将其放入另一组数据集
//fill_queue表示data1中的第i个将被填入data0中的fill_queue[i]的数据
void DataPreparer::transData(const Matrix& X0, const Matrix& Y0, Matrix& X1, Matrix& Y1, const std::vector<int>& fill_queue)
{
    if (shuffle_ == 0 && trans_ == 0)
    {
        Matrix::copyData(X0, X1);
        Matrix::copyData(Y0, Y1);
    }
    else
    {
#pragma omp parallel
        {
            Matrix x = X1.cloneSharedCol(0), y = Y1.cloneSharedCol(0);
#pragma omp for
            for (int i = 0; i < fill_queue.size(); i++)
            {
                //首先复制数据
                if (shuffle_)
                {
                    Matrix::copyDataPointer(X0, X0.getDataPointer(0, fill_queue[i]), X1, X1.getDataPointer(0, i), X1.getRow());
                    Matrix::copyDataPointer(Y0, Y0.getDataPointer(0, fill_queue[i]), Y1, Y1.getDataPointer(0, i), Y1.getRow());
                }
#ifndef _DEBUG
                if (trans_)
                {
                    //注意此处反复构造和析构矩阵，如果使用单线程会拖慢速度，DEBUG模式下不进行此操作
                    x.shareData(X1, 0, i);
                    y.shareData(Y1, 0, i);
                    transOne(x, y);
                }
#else
                //LOG("Warning: never transfer images on Debug mode!\n");
#endif
            }
        }
    }
}

void DataPreparer::shuffleQueue(std::vector<int>& train_queue)
{
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(train_queue.begin(), train_queue.end(), g);
}

void DataPreparer::reload()
{
    //safe_delete(train_data_preparer_);
    //train_data_preparer_ = DataPreparer::createByOption(option_);
}

//数据准备器将origin中的数据打乱，进行变换之后上传至data
void DataPreparer::prepareData(int epoch, const std::string& info, Matrix& X_origin, Matrix& Y_origin, Matrix& X_cpu, Matrix& Y_cpu)
{
    int ep1 = epoch + 1;

    if (train_queue_origin_.size() != X_origin.getNumber())
    {
        train_queue_origin_.resize(X_origin.getNumber());
        for (int i = 0; i < train_queue_origin_.size(); i++)
        {
            train_queue_origin_[i] = i;
        }
    }

    std::vector<int> train_queue_cpu(X_cpu.getNumber());
    bool need_refill = false;
    //复制训练序列origin到cpu，注意这种处理方式是考虑了二者尺寸可能不同的情况
    int i = 0;
    while (i < train_queue_cpu.size())
    {
        if (trained_in_origin_ >= train_queue_origin_.size())
        {
            trained_in_origin_ = 0;
        }
        //如果出现越界就说明需要处理下一循环
        if (trained_in_origin_ == 0)
        {
            if (shuffle_)
            {
                //LOG("Shuffle queue for epoch %d\n", ep1);
                shuffleQueue(train_queue_origin_);
            }
            need_refill = true;
        }
        train_queue_cpu[i] = train_queue_origin_[trained_in_origin_];
        i++;
        trained_in_origin_++;
    }
    //重新采集train，其实严格说应该在循环内部采集，因为变换在后面
    if (need_refill && fill_)
    {
        Log::LOG("Fill data for epoch %d\n", ep1);
        fillData(X_origin, Y_origin);
    }

    //可能包含对训练数据的变换
    Log::LOG("Transfer data for epoch %d\n", ep1);
    transData(X_origin, Y_origin, X_cpu, Y_cpu, train_queue_cpu);
}

void DataPreparer::initData(Matrix& X, Matrix& Y)
{
    if (!Option::getInstance().hasSection(section_))
    {
        return;
    }
    assert(!X.inGPU() && !Y.inGPU());
    //通常此时data是空的，先按照网络的输入和输出尺寸修改数据的维度，组数为0不变
    resetDataDim(X, Y);
    //以下需注意数据集的维度和组数有可能在读取时被改变
    if (fill_ == 0)
    {
        if (Option::getInstance().getInt(section_, "data_in_txt", 0))
        {
            readTxt(Option::getInstance().getString(section_, "data_file"), X, Y);
        }
        else
        {
            readBin(Option::getInstance().getString(section_, "x_file"), Option::getInstance().getString(section_, "y_file"), X, Y);
        }
    }
    else
    {
        fillData(X, Y);
        int force_train_group = getFillGroup();
        if (force_train_group > 0)
        {
            X.resizeNumber(force_train_group);
            Y.resizeNumber(force_train_group);
        }
    }
}

//读取数据
void DataPreparer::initData(Matrix& X, Matrix& Y, Matrix& X_cpu, Matrix& Y_cpu)
{
    initData(X, Y);
    X_cpu.resize(X);
    Y_cpu.resize(Y);
}

//从txt文件读取数据到DataGroup
//该函数只读取到CPU，如需读取至GPU请调用后再写一步
//这里的处理可能不是很好
void DataPreparer::readTxt(const std::string& filename, Matrix& X, Matrix& Y)
{
    int count = 0;
    if (filename == "")
    {
        //setDataDim(data);
        return;
    }

    int mark = 3;
    //数据格式：前两个是输入变量数和输出变量数，之后依次是每组的输入和输出，是否有回车不重要
    std::string str = convert::readStringFromFile(filename);
    if (str == "")
    {
        return;
    }
    std::vector<real> v;
    int n = convert::findNumbers(str, v);
    if (n <= 0)
    {
        return;
    }

    int x_size = int(v[0]);
    int y_size = int(v[1]);

    count = (n - mark) / (x_size + y_size);
    X.resize(x_size, count);
    Y.resize(y_size, count);

    //写法太难看了
    int k = mark, k1 = 0, k2 = 0;

    for (int i_data = 1; i_data <= count; i_data++)
    {
        for (int i = 1; i <= x_size; i++)
        {
            X.getData(k1++) = v[k++];
        }
        for (int i = 1; i <= y_size; i++)
        {
            Y.getData(k2++) = v[k++];
        }
    }
}

//从bin文件读取数据到DataGroup
//该函数只读取到CPU，如需读取至GPU请调用后再写一步
void DataPreparer::readBin(const std::string& file_bin_x, const std::string& file_bin_y, Matrix& X, Matrix& Y)
{
    readBin(file_bin_x, X);
    readBin(file_bin_y, Y);
}

void DataPreparer::readBin(const std::string& file_bin, Matrix& data)
{
    if (!File::fileExist(file_bin))
    {
        return;
    }
    auto data_bin = File::readFile(file_bin.c_str());
    //二进制数据文件定义：宽，高，通道，图片数，数据
    int w = *(int*)(data_bin.data());
    int h = *(int*)(data_bin.data() + 4);
    int c = *(int*)(data_bin.data() + 8);
    int n = *(int*)(data_bin.data() + 12);
    Log::LOG("Read bin file, [%d, %d, %d, %d]\n", w, h, c, n);

    int64_t total_size = w * h * c * n;
    data.resize(w, h, c, n);
    for (int64_t i = 0; i < total_size; i++)
    {
        auto v = *(real*)(data_bin.data() + 16 + i * sizeof(real));
        data.getData(i) = v;
    }
}

void DataPreparer::writeBin(const std::string& file_bin, const Matrix& data)
{
    std::vector<char> data_bin(16 + data.getDataSizeInByte());
    *(int*)(data_bin.data()) = data.getWidth();
    *(int*)(data_bin.data() + 4) = data.getHeight();
    *(int*)(data_bin.data() + 8) = data.getChannel();
    *(int*)(data_bin.data() + 12) = data.getNumber();
    memcpy(data_bin.data() + 16, data.getDataPointer(), data.getDataSizeInByte());
    File::writeFile(file_bin, data_bin.data(), data_bin.size());
}

//若读取数据失败，依据网络的输入和输出创建一个，避免后续的错误，无太大的意义
void DataPreparer::resetDataDim(Matrix& X, Matrix& Y)
{
    X.resizeKeepNumber(dimx_);
    Y.resizeKeepNumber(dimy_);
}

std::string DataPreparer::getMessage(int i)
{
    if (i >= 0 && i < message_.size())
    {
        return message_[i];
    }
    return "Error.";
}

}    // namespace woco