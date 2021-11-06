#pragma once
#include "DataPreparerFactory.h"
#include "Net.h"
#include "Timer.h"
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace cccc
{

//总调度
class Brain : public Neural
{
public:
    Brain();
    virtual ~Brain();

public:
    int brain_id_;

protected:
    int batch_;
    std::function<void(Brain*)> running_callback_ = nullptr;    //回调函数
    Option option_;
    Timer timer_total_;
    int MP_count_ = 1;
    std::vector<std::unique_ptr<Net>> nets_;
    WorkModeType work_mode_ = WORK_MODE_NORMAL;

public:
    void setCallback(std::function<void(Brain*)> f) { running_callback_ = f; }
    Option* getOption() { return &option_; }
    void setOption(Option* op)
    {
        option_ = *op;
    }

protected:
    //epoch和iter的统计
    int epoch_count_ = 0;
    int iter_count_ = 0;

public:
    int getIterCount() { return iter_count_; }
    void setIterCount(int ic) { iter_count_ = ic; }

public:
    //real scale_data_x_ = 1;   //x参与计算之前先乘以此数，太麻烦，不搞了
    //real train_accuracy_ = 0;    //训练集上准确率
    //real test_accuracy_ = 0;     //测试集上准确率

public:
    //训练集
    //这里采取的是变换之后将数据传入显存的办法，每个epoch都需要有这一步
    //这里理论上会降低速度，如果在显存里保存两份数据交替使用可以增加效率
    //此处并未使用，给GPU一些休息的时间

    //原始数据组数为cpu的数据的整数倍，通常为1倍，准备器中train_queue的变换保证数据被选中的机会一致
    //如果每次原始数据都是即时生成，算法并无变化，但最好设置为等同于cpu数据组数，以使数据充分被利用

    //原始训练集
    Matrix X_train_, Y_train_;
    //经变换后的训练集，与上面的区别是顺序被随机打乱，以及可能增加了一定的干扰，其后会被直接上传至gpu
    Matrix X_train_cpu_, Y_train_cpu_;
    //原始测试集
    Matrix X_test_, Y_test_;
    //经变换后的测试集，不同之处同训练集
    Matrix X_test_cpu_, Y_test_cpu_;

    DataPreparerFactory::UniquePtr data_preparer_;    //训练集准备器
    DataPreparerFactory::UniquePtr& getDataPreparer() { return data_preparer_; }

    DataPreparerFactory::UniquePtr data_preparer2_;    //测试集准备器

public:
    //此处需注意顺序
    //nets初始化之后如不修改出入的数据结构，则可以再次初始化，即在半途更换网络结构，求解器等操作可以实现
    int init(const std::string& ini = "");
    int loadIni(const std::string& ini = "");
    int testGPUDevice();
    int initNets();
    int initData();

    virtual void initDataPreparer();
    void initTrainData();
    void initTestData();

public:
    void train(std::vector<std::unique_ptr<Net>>& nets, DataPreparer* data_preparer, int epochs);
    //void train(std::vector<std::unique_ptr<Net>>& net, DataPreparer* data_preparer, int epochs) { train({ net }, data_preparer, epochs); }

private:
    //在主线程进行数据准备，副线程进行训练和测试
    struct TrainInfo
    {
        std::atomic<int> data_prepared;           //0 未准备好，1 cpu准备完毕， 2 gpu准备完毕
        std::atomic<int> data_distributed;        //已经复制过的线程
        std::atomic<int> stop;                    //结束信息
        std::atomic<int> trained;                 //已经训练完成的网络个数
        std::atomic<int> parameters_collected;    //数据同步完毕
        void reset()
        {
            data_prepared = 0;
            data_distributed = 0;
            stop = 0;
            trained = 0;
            parameters_collected = 0;
        }
        TrainInfo() { reset(); }
        ~TrainInfo() {}
    };
    void trainOneNet(std::vector<std::unique_ptr<Net>>& nets, int net_id, TrainInfo& train_info, int epoch0, int epochs);

public:
    //获取网络的时候，应注意当前的gpu
    Net* getNet(int i = 0)
    {
        if (nets_.size() > i)
        {
            return nets_[i].get();
        }
        return nullptr;
    }
    const std::vector<std::unique_ptr<Net>>& getNets() { return nets_; }

public:
    //以下构成一组调度范例
    void run(int train_epochs = -1);
    void testData(Net* net, int force_output = 0, int test_type = 0);
    void extraTest(Net* net, const std::string& section, int force_output = 0, int test_type = 0);

public:
    int testExternalData(void* x, void* y, void* a, int n, int attack_times = 0, realc* error = nullptr);

public:
    void setLearnRateBase(real lrb)
    {
        for (auto& n : nets_)
        {
            n->setLearnRateBase(lrb);
        }
    }
};

}    // namespace cccc