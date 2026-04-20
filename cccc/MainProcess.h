#pragma once
#include "DataPreparerFactory.h"
#include "Net.h"
#include "Timer.h"
#include <atomic>
#include <vector>

namespace cccc
{

//总调度
class DLL_EXPORT MainProcess
{
public:
    MainProcess();
    virtual ~MainProcess();
    MainProcess(const MainProcess&) = delete;
    MainProcess& operator=(const MainProcess&) = delete;

public:
    int brain_id_ = 0;

protected:
    int batch_ = 0;
    std::function<void(MainProcess*)> running_callback_ = nullptr;    //回调函数
    Option option_;
    Timer timer_total_;
    int MP_count_ = 1;
    std::string train_filename_;
    std::vector<GpuControl> gpus_;    //需要在网络后面析构，即GpuControl必须在所有矩阵析构之后析构
    std::vector<std::unique_ptr<Net>> nets_;

    WorkModeType work_mode_ = WORK_MODE_NORMAL;

public:
    void setCallback(std::function<void(MainProcess*)> f) { running_callback_ = f; }

    Option* getOption() { return &option_; }
    void setTrainFilename(const std::string& filename) { train_filename_ = filename; }    //训练集文件名，主要用于保存权重时的命名

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
    std::string makeSaveName(const std::string& save_format, int epoch_count, bool is_filename = true);
    std::string makeSaveSign();

public:
    //以下构成一组调度范例
    void train(std::vector<std::unique_ptr<Net>>& nets, DataPreparer* data_preparer, int total_epochs);
    //void train(std::vector<std::unique_ptr<Net>>& net, DataPreparer* data_preparer, int epochs) { train({ net }, data_preparer, epochs); }

private:
    //在主线程进行数据准备，副线程进行训练和测试
    struct TrainInfo
    {
        std::atomic<int> data_prepared;          //0 未准备好，1 cpu准备完毕， 2 gpu准备完毕
        std::atomic<int> data_distributed;       //复制完数据的线程数
        std::atomic<int> stop;                   //结束信息
        std::atomic<int> trained;                //已经训练完成的网络个数
        std::atomic<int> dweight_uncollected;    //梯度同步完毕

        //std::atomic<int> need_reset;             //需要重置求解器的信号

        TrainInfo() { reset(); }

        void reset()
        {
            data_prepared = 0;
            data_distributed = 0;
            stop = 0;
            trained = 0;
            dweight_uncollected = 0;
        }
    };

    void trainOneNet(std::vector<std::unique_ptr<Net>>& nets, int net_id, TrainInfo& ti, int total_epochs);

    static bool checkTestEffective(std::vector<TestInfo>& group_result_max, const std::vector<TestInfo>& group_result);
    static bool checkTrainHealth(const std::vector<TestInfo>& group_result_prev, const std::vector<TestInfo>& group_result, float l1p, float l2p, float l1, float l2, double increase_limited, int effective_epoch_count);

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
    void run();
    void testData(Net* net, int epoch = 0, int total_epochs = 0, int* max_accuracy_epoch = nullptr, float* max_accuracy = nullptr, std::vector<std::vector<TestInfo>>* collect_test_info = nullptr, Matrix* X = nullptr, Matrix* Y = nullptr);
    void extraTest(Net* net, const std::string& section, int force_output = 0, int test_type = 0);

public:
    int testExternalData(void* x, void* y, void* a, int n, int attack_times = 0, double* error = nullptr);
};

}    // namespace cccc