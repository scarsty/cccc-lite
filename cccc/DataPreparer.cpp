#include "DataPreparer.h"

#include "Timer.h"
#include "filefunc.h"
#include <ctime>
#include <thread>

namespace cccc
{

DataPreparer::DataPreparer() :
    X(DataType::CURRENT, UnitType::CPU),
    Y(DataType::CURRENT, UnitType::CPU),
    X0(DataType::CURRENT, UnitType::CPU),
    Y0(DataType::CURRENT, UnitType::CPU)
{
    //rand_.set_seed();
    rand_.set_parameter(0, 1);
}

DataPreparer::~DataPreparer()
{
}

void DataPreparer::init()
{
    OPTION_GET_INT(shuffle_);
    OPTION_GET_INT(trans_);
    if (create_by_dll_ != "")
    {
        fill_ = 1;
    }
    OPTION_GET_INT(fill_);
    OPTION_GET_INT(fill_group_);
    OPTION_GET_INT(fill_period_);
    init2();
    LOG("Initialize dataset\n");
    resetDataDim();
    initData();
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

void DataPreparer::addMatrix(const std::string& name, const std::vector<int>& dim)
{
    extra_data_.insert({ name, Matrix(dim, DataType::CURRENT, UnitType::CPU) });
    extra_data0_.insert({ name, Matrix(dim, DataType::CURRENT, UnitType::CPU) });
}

void DataPreparer::initData()
{
    X.resizeNumber(fill_group_);
    Y.resizeNumber(fill_group_);
    X0.resizeNumber(fill_group_);
    Y0.resizeNumber(fill_group_);

    queue_origin_.resize(X.getNumber());
    for (int i = 0; i < queue_origin_.size(); i++)
    {
        queue_origin_[i] = i;
    }
}

void DataPreparer::shuffleQueue(std::vector<int>& train_queue)
{
    std::ranges::shuffle(train_queue, std::mt19937(std::random_device()()));
}

//数据准备器将原始数据打乱，进行变换之后上传至data
//对于验证来说，这里前面有些拖沓
void DataPreparer::prepareData(int epoch, const std::string& info)
{
    Timer timer;
    //重新采集未变换的数据

    if (fill_ && fill_period_ > 0 && epoch % fill_period_ == 0)
    {
        //LOG("Data for epoch {} Fill\n", epoch + 1);
        fillData0();
        is_new_data_ = true;
    }
    else
    {
        is_new_data_ = false;
        return;
    }

    int concurrency = std::thread::hardware_concurrency() / 2;

    //可能包含对训练数据的变换
    if (X.getDataPtr() != X0.getDataPtr() && Y.getDataPtr() != Y0.getDataPtr())
    {
        if (shuffle_ == 0)
        {
            Matrix::copyData(X0, X);
            Matrix::copyData(Y0, Y);
            for (auto& [name, m] : extra_data0_)
            {
                Matrix::copyData(m, extra_data_[name]);
            }
        }
        else
        {
            LOG("Shuffle data for epoch {}\n", epoch + 1);
            shuffleQueue(queue_origin_);
            std::vector<std::thread> threads(concurrency);
            int count_th = (queue_origin_.size() + concurrency - 1) / concurrency;
            int i_th = 0;
            for (auto& t : threads)
            {
                t = std::thread{ [i_th, count_th, this]()
                    {
                        auto x = X0.createSharedCol(0), y = Y0.createSharedCol(0);
                        auto dim = x.getDim();
                        for (int i = i_th * count_th; i < std::min(i_th * count_th + count_th, int(queue_origin_.size())); i++)
                        {
                            Matrix::copyRows(X0, queue_origin_[i], X, i, 1);
                            Matrix::copyRows(Y0, queue_origin_[i], Y, i, 1);
                            for (auto& [name, m] : extra_data0_)
                            {
                                Matrix::copyRows(m, queue_origin_[i], extra_data_[name], i, 1);
                            }
                            if (trans_)
                            {
                                x.shareData(X, 0, i);
                                y.shareData(Y, 0, i);
                                transOne(x, y);
                            }
                        }
                    } };
                i_th++;
            }
            for (auto& t : threads)
            {
                t.join();
            }
        }
    }
    //X_gpu.resize(X.getDim());
    //Y_gpu.resize(Y.getDim());
    //Matrix::copyData(X, X_gpu);
    //Matrix::copyData(Y, Y_gpu);
    LOG("Data for epoch {} prepared in {} s\n", epoch + 1, timer.getElapsedTime());
}

//依据网络的输入和输出创建一个，避免可能的失败
void DataPreparer::resetDataDim()
{
    X.resizeKeepNumber(dim0_);
    Y.resizeKeepNumber(dim1_);
    X0.resizeKeepNumber(dim0_);
    Y0.resizeKeepNumber(dim1_);
    //X_gpu = Matrix(dim0_, DataType::CURRENT, UnitType::GPU);
    //Y_gpu = Matrix(dim1_, DataType::CURRENT, UnitType::GPU);
}

std::string DataPreparer::getMessage(int i)
{
    if (i >= 0 && i < message_.size())
    {
        return message_[i];
    }
    return "Error.";
}

}    // namespace cccc