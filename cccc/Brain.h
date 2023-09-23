#pragma once
#include "DataPreparerFactory.h"
#include "Net.h"
#include "Timer.h"
#include <atomic>
#include <vector>

namespace cccc
{

//总调度
class DLL_EXPORT Brain
{
public:
    Brain();
    virtual ~Brain();
    Brain(const Brain&) = delete;
    Brain& operator=(const Brain&) = delete;

public:
    int brain_id_;

protected:
    int batch_;
    std::function<void(Brain*)> running_callback_ = nullptr;    //回调函数
    Option option_;
    Timer timer_total_;
    int MP_count_ = 1;
    std::vector<std::unique_ptr<Net>> nets_;
    std::vector<GpuControl> gpus_;
    WorkModeType work_mode_ = WORK_MODE_NORMAL;

public:
    void setCallback(std::function<void(Brain*)> f) { running_callback_ = f; }
    Option* getOption() { return &option_; }

protected:
    //epoch和iter的统计
    int epoch_count_ = 0;
    int iter_count_ = 0;

public:
    int getIterCount() { return iter_count_; }
    void setIterCount(int ic) { iter_count_ = ic; }

public:
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
    std::string makeSaveSign();

public:
    //以下构成一组调度范例
    void train(std::vector<std::unique_ptr<Net>>& nets, DataPreparer* data_preparer, int total_epochs);
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
        std::atomic<int> need_reset;              //需要重置求解器的信号
        void reset()
        {
            data_prepared = 0;
            data_distributed = 0;
            stop = 0;
            trained = 0;
            parameters_collected = 0;
        }
        TrainInfo() { reset(); }
    };
    void trainOneNet(std::vector<std::unique_ptr<Net>>& nets, int net_id, TrainInfo& train_info, int total_epochs);

public:
    //获取网络的时候，应注意当前的gpu
    Net* getNet(int i = 0) const
    {
        if (nets_.size() > i)
        {
            return nets_[i].get();
        }
        return nullptr;
    }
    const std::vector<std::unique_ptr<Net>>& getNets() { return nets_; }

public:
    void run(int train_epochs = -1);
    void testData(Net* net, int force_output = 0, int test_type = 0);
    void extraTest(Net* net, const std::string& section, int force_output = 0, int test_type = 0);

public:
    int testExternalData(void* x, void* y, void* a, int n, int attack_times = 0, realc* error = nullptr);
};

}    // namespace cccc