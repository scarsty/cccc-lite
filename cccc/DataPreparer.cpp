#include "DataPreparer.h"
#include "filefunc.h"
#include <ctime>
#include <thread>

namespace cccc
{

DataPreparer::DataPreparer()
    : X(UnitType::CPU), Y(UnitType::CPU), X0(UnitType::CPU), Y0(UnitType::CPU), LW(UnitType::CPU), LW0(UnitType::CPU)
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
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(train_queue.begin(), train_queue.end(), g);
}

//数据准备器将原始数据打乱，进行变换之后上传至data
//对于验证来说，这里前面有些拖沓
void DataPreparer::prepareData(int epoch, const std::string& info)
{
    //重新采集未变换的数据
    if (fill_)
    {
        LOG("Fill data for epoch {}\n", epoch + 1);
        fillData0();
    }

    int concurrency = std::thread::hardware_concurrency() / 2;

    //可能包含对训练数据的变换
    if (X.getDataPtr() != X0.getDataPtr() && Y.getDataPtr() != Y0.getDataPtr())
    {
        if (shuffle_ == 0)
        {
            Matrix::copyData(X0, X);
            Matrix::copyData(Y0, Y);
            Matrix::copyData(LW0, LW);
        }
        else
        {
            LOG("Shuffle data for epoch {}\n", epoch + 1);
            shuffleQueue(queue_origin_);

            //#pragma omp parallel
            //                {
            //                    auto x = X0.cloneSharedCol(0), y = Y0.cloneSharedCol(0);
            //                    auto dim = x.getDim();
            //                    //std::swap(dim[dim.size() - 1], dim[dim.size() - 2]);
            //                    //x.resize(dim);
            //                    //dim = y.getDim();
            //                    //std::swap(dim[dim.size() - 1], dim[dim.size() - 2]);
            //                    //y.resize(dim);
            //#pragma omp for
            //                    for (int i = 0; i < queue_origin_.size(); i++)
            //                    {
            //                        Matrix::copyRows(X0, queue_origin_[i], X, i, 1);
            //                        Matrix::copyRows(Y0, queue_origin_[i], Y, i, 1);
            //                        Matrix::copyRows(LW0, queue_origin_[i], LW, i, 1);
            //                        x.shareData(X, 0, i);
            //                        y.shareData(Y, 0, i);
            //                        transOne(x, y);
            //                    }
            //                }
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
                            Matrix::copyRows(LW0, queue_origin_[i], LW, i, 1);
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
    LOG("Data prepared for epoch {}\n", epoch + 1);
}

//依据网络的输入和输出创建一个，避免可能的失败
void DataPreparer::resetDataDim()
{
    X.resizeKeepNumber(dim0_);
    Y.resizeKeepNumber(dim1_);
    X0.resizeKeepNumber(dim0_);
    Y0.resizeKeepNumber(dim1_);
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