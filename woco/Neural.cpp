#include "Neural.h"
#include "ConsoleControl.h"
#include "Factory.h"
#include "File.h"
#include "Log.h"
#include "Option.h"
#include "Random.h"
#include "convert.h"
#include <algorithm>

namespace woco
{

//C++14可以写在定义，目前swig有问题
Neural::Neural() : X_train_{ DeviceType::CPU }, Y_train_{ DeviceType::CPU }, X_train_cpu_{ DeviceType::CPU }, Y_train_cpu_{ DeviceType::CPU }, X_test_{ DeviceType::CPU }, Y_test_{ DeviceType::CPU }, X_test_cpu_{ DeviceType::CPU }, Y_test_cpu_{ DeviceType::CPU }
{
}

//返回为0是正确创建
int Neural::init(const std::string& ini)
{
    if (loadIni(ini))
    {
        Log::LOG("Error: Load ini file failed!!\n");
        return 1;
    }
    if (testGPUDevice())
    {
        Log::LOG("Error: GPU state is not right!!\n");
        return 2;
    }
    if (initNets())
    {
        Log::LOG("Error: Initial net(s) failed!!\n");
        return 3;
    }
    if (initData())
    {
        Log::LOG("Error: Initial data failed!!\n");
        return 4;
    }
    return 0;
}

int Neural::loadIni(const std::string& ini)
{
    Log::LOG("%s\n", Timer::getNowAsString().c_str());

    //Log::LOG("Size of real is %lu bytes\n", sizeof(real));

    //初始化选项
    //貌似这个设计比较瞎
    if (ini != "")
    {
        if (File::fileExist(ini))
        {
            Option::getInstance().loadIniFile(ini);
        }
        else
        {
            Option::getInstance().loadIniString(ini);
        }
    }
    //Option::getInstance().print();
    //batch_ = (std::max)(1, Option::getInstance().getInt("", "batch", 100));
    //work_mode_ = Option::getInstance().getEnum("", "work_mode", WORK_MODE_NORMAL);

    return 0;
}

int Neural::testGPUDevice()
{
    //gpu测试
    int device_count = 0;
    if (Option::getInstance().getInt("", "use_cuda", 1))
    {
        device_count = CudaControl::checkDevices();
        if (device_count > 0)
        {
            Log::LOG("Found %d CUDA device(s)\n", device_count);
            CudaControl::setGlobalCudaType(DeviceType::GPU);
        }
        else
        {
            Log::LOG("Error: No CUDA devices!!\n");
            CudaControl::setGlobalCudaType(DeviceType::CPU);
            return 1;
        }
    }

    if (Option::getInstance().getInt("", "use_cuda") != 0 && CudaControl::getGlobalCudaType() != DeviceType::GPU)
    {
        Log::LOG("CUDA state is not right, refuse to run!\n");
        Log::LOG("Re-init the net again, or consider CPU mode (slow).\n");
        return 1;
    }

    MP_count_ = (std::min)(device_count, Option::getInstance().getInt("", "mp", 1));
    if (MP_count_ <= 0)
    {
        MP_count_ = 1;
    }
    return 0;
}

int Neural::initNets()
{
    nets_.resize(MP_count_);
    //这里读取ini中指定的device顺序，其中第一个设备为主网络，该值一般来说应由用户指定
    auto mp_device = Option::getInstance().getVector<int>("", "mp_device");
    //如果用户指定的不正确，则以best_device的顺序决定
    auto check_repeat = [](std::vector<int> v)
    {
        for (int i = 0; i < int(v.size()) - 1; i++)
        {
            for (int j = i + 1; j < v.size(); j++)
            {
                if (v[i] == v[j])
                {
                    return true;
                }
            }
        }
        return false;
    };
    if (mp_device.size() < MP_count_ || check_repeat(mp_device))
    {
        mp_device.resize(MP_count_);
        for (int i = 0; i < MP_count_; i++)
        {
            mp_device[i] = CudaControl::getBestDevice(i);
        }
    }
    for (int i = 0; i < MP_count_; i++)
    {
        CudaControl::select(mp_device[i]);    //需在创建网络之前
        auto& net = nets_[i] = Factory::createNet();
        //net->setDevice(mp_device[i]);
        int dev_id = net->getDevice();
        if (dev_id >= 0)
        {
            Log::LOG("Net %d will be created on device %d\n", i, dev_id);
        }
        else
        {
            Log::LOG("Net %d will be created on CPU\n", i);
        }
        CudaControl::select(mp_device[i]);
        //net.setOption(option_);
        //net.setBatch(batch_ / MP_count_);
        //if (net->init() != 0)
        //{
        //    return 1;
        //}
        net->makeStructure();
    }

    //合并网络的权值，方便数据交换
    if (MP_count_ > 1)
    {
        for (int i = 0; i < nets_.size(); i++)
        {
            Log::LOG("Combine parameters of net %d\n", i);
            auto& net = nets_[i];
            net->combineParameters();
            //只使用0号网络的权值
            if (i != 0)
            {
                Matrix::copyDataAcrossDevice(nets_[0]->getCombinedWeight(), net->getCombinedWeight());
            }
        }
    }

    //主线程使用0号网络
    nets_[0]->setDeviceSelf();
    return 0;
}

int Neural::initData()
{
    initDataPreparer();
    initTrainData();
    initTestData();
    return 0;
}

void Neural::initDataPreparer()
{
    //数据准备器
    //这里使用公共数据准备器，实际上完全可以创建私有的准备器
    auto dim0 = nets_[0]->X().getDim();
    auto dim1 = nets_[0]->Y().getDim();
    //DataPreparerFactory::destroy(data_preparer_);
    dp_train_ = Factory::createDP("data_train", dim0, dim1);
    std::string test_section = "data_test";
    if (!Option::getInstance().hasSection(test_section))
    {
        Option::getInstance().setOption("", "test_test", "0");
        Option::getInstance().setOption("", "test_test_origin", "0");
        return;
        //Option::getInstance().setOption(test_section, "test", "1");
    }
    dp_test_ = Factory::createDP(test_section, dim0, dim1);
}

//初始化训练集，必须在DataPreparer之后
void Neural::initTrainData()
{
    dp_train_->initData(X_train_, Y_train_, X_train_cpu_, Y_train_cpu_);
}

//生成测试集
void Neural::initTestData()
{
    if (dp_test_)
    {
        dp_test_->initData(X_test_, Y_test_, X_test_cpu_, Y_test_cpu_);
        dp_test_->prepareData(0, "test", X_test_, Y_test_, X_test_cpu_, Y_test_cpu_);
    }
}

//运行，注意容错保护较弱
//注意通常情况下是使用第一个网络测试数据
void Neural::run(int train_epochs /*= -1*/)
{
    auto net = nets_[0];
    //初测
    testData(net, Option::getInstance().getInt("", "force_output"), Option::getInstance().getInt("", "test_max"));

    if (train_epochs < 0)
    {
        train_epochs = Option::getInstance().getInt("", "train_epochs", 20);
    }
    Log::LOG("Running for %d epochs...\n", train_epochs);

    train(nets_, dp_train_, train_epochs);

    std::string save_filename = Option::getInstance().getString("", "SaveFile");
    if (save_filename != "")
    {
        net->saveWeight(save_filename);
    }

    //终测
    testData(net, Option::getInstance().getInt("", "force_output"), Option::getInstance().getInt("", "test_max"));
    //附加测试，有多少个都能用
    extraTest(net, "extra_test", Option::getInstance().getInt("", "force_output"), Option::getInstance().getInt("", "test_max"));

#ifdef LAYER_TIME
    for (auto& l : nets_[0]->getLayerVector())
    {
        Log::LOG("%s: %g,%g,%g\n", l->getName().c_str(), l->total_time1_, l->total_time2_, l->total_time3_);
    }
#endif

    auto time_sec = timer_total_.getElapsedTime();
    net->save("");
    Log::LOG("Run neural net end. Total time is %s\n", Timer::autoFormatTime(time_sec).c_str());
    Log::LOG("%s\n", Timer::getNowAsString().c_str());

    CudaControl::destroyAll();
}

//训练一批数据，输出步数和误差，若训练次数为0可以理解为纯测试模式
//首个参数为指定几个结构完全相同的网络并行训练
void Neural::train(std::vector<Net*>& nets, DataPreparer* data_preparer, int epochs)
{
    if (epochs <= 0)
    {
        return;
    }

    int iter_per_epoch = X_train_.getNumber() / nets_[0]->X().getNumber();    //如果不能整除，则估计会不准确，但是关系不大
    if (iter_per_epoch <= 0)
    {
        iter_per_epoch = 1;
    }
    epoch_count_ = iter_count_ / iter_per_epoch;

    real e = 0, e0 = 0;

    Timer timer_per_epoch, timer_trained;
    //prepareData();
    TrainInfo train_info;
    train_info.data_prepared = 0;

    int test_test = Option::getInstance().getInt("", "test_test", 0);
    int test_test_origin = Option::getInstance().getInt("", "test_test_origin", 0);
    int test_epoch = Option::getInstance().getInt("", "test_epoch", 1);

    //创建训练进程
    std::vector<std::thread> net_threads;
    for (int i = 0; i < nets_.size(); i++)
    {
        net_threads.emplace_back(std::thread{ [this, &nets, i, &train_info, &epochs]()
            { trainOneNet(nets, i, train_info, epoch_count_, epochs); } });
    }

    train_info.stop = 0;
    int epoch0 = epoch_count_;
    for (int epoch_count = epoch0; epoch_count < epoch0 + epochs; epoch_count++)
    {
        data_preparer->prepareData(epoch_count, "train", X_train_, Y_train_, X_train_cpu_, Y_train_cpu_);
        if (dp_test_ && (test_test || test_test_origin) && epoch_count % test_epoch == 0 && dp_test_->isFill())
        {
            dp_test_->prepareData(epoch_count, "test", X_test_, Y_test_, X_test_cpu_, Y_test_cpu_);
        }
        train_info.data_prepared = 1;
        WAIT_UNTIL(train_info.data_distributed == MP_count_);
        train_info.data_prepared = 0;
        train_info.data_distributed = 0;
        iter_count_ += iter_per_epoch;
        epoch_count_++;
        if (epoch_count - epoch0 > 0)
        {
            double left_time = timer_trained.getElapsedTime() / (epoch_count - epoch0) * (epoch0 + epochs - epoch_count);    //注意这个估测未考虑额外测试的消耗，不是很准
            Log::LOG("%g s for this epoch, %s elapsed totally, about %s left\n",
                timer_per_epoch.getElapsedTime(), Timer::autoFormatTime(timer_total_.getElapsedTime()).c_str(), Timer::autoFormatTime(left_time).c_str());
        }
        timer_per_epoch.start();
        if (train_info.stop)
        {
            break;
        }
    }
    //train_info.stop = 1;

    for (int i = 0; i < net_threads.size(); i++)
    {
        net_threads[i].join();
    }
    //delete_all(net_threads);

    Log::LOG("%g s for this epoch, %s elapsed totally\n", timer_per_epoch.getElapsedTime(), Timer::autoFormatTime(timer_total_.getElapsedTime()).c_str());
    Log::LOG("\n");
}

//训练网络数组nets中的一个
void Neural::trainOneNet(std::vector<Net*>& nets, int net_id, TrainInfo& train_info, int epoch0, int epochs)
{
    auto& net = nets[net_id];
    net->setDeviceSelf();
    net->setActivePhase(ACTIVE_PHASE_TRAIN);

    Matrix X_train_gpu(DeviceType::GPU), Y_train_gpu(DeviceType::GPU), A_train_gpu(DeviceType::GPU);    //train_cpu的映像，如有多个GPU，每个是其中的一部分
    X_train_gpu.resizeAndNumber(X_train_cpu_.getDim(), X_train_cpu_.getNumber() / nets.size());
    Y_train_gpu.resizeAndNumber(Y_train_cpu_.getDim(), Y_train_cpu_.getNumber() / nets.size());
    auto& X_train_sub = net->X();
    auto& Y_train_sub = net->Y();
    auto& A_train_sub = net->A();

    Matrix X_test_gpu(DeviceType::GPU), Y_test_gpu(DeviceType::GPU), A_test_gpu(DeviceType::GPU);
    if (net_id == 0)
    {
        X_test_gpu.resize(X_test_cpu_.getDim());
        Matrix::copyData(X_test_cpu_, X_test_gpu);
        Y_test_gpu.resize(Y_test_cpu_.getDim());
        Matrix::copyData(Y_test_cpu_, Y_test_gpu);
    }

    int test_train = Option::getInstance().getInt("", "test_train", 0);
    int test_train_origin = Option::getInstance().getInt("", "test_train_origin", 0);
    int pre_test_train = Option::getInstance().getInt("", "pre_test_train", 0);
    int test_test = Option::getInstance().getInt("", "test_test", 0);
    int test_test_origin = Option::getInstance().getInt("", "test_test_origin", 0);
    int test_epoch = Option::getInstance().getInt("", "test_epoch", 1);
    int save_epoch = Option::getInstance().getInt("", "save_epoch", 10);
    int out_iter = Option::getInstance().getInt("", "out_iter", 100);
    int test_max = Option::getInstance().getInt("", "test_max", 0);
    std::string save_format = Option::getInstance().getString("", "save_format", "save/save-{epoch}.txt");
    int total_batch = X_train_cpu_.getNumber();

    realc max_test_origin_accuracy = 0;
    int max_test_origin_accuracy_epoch = 0;
    realc max_test_accuracy = 0;
    int max_test_accuracy_epoch = 0;

    int iter_count = 0;
    int epoch_count = epoch0;
    while (epoch_count < epoch0 + epochs)
    {
        epoch_count++;
        //等待数据准备完成
        WAIT_UNTIL(train_info.data_prepared == 1);

        Matrix::copyDataPointer(X_train_cpu_, X_train_cpu_.getDataPointer(0, 0, 0, net_id * X_train_gpu.getNumber()), X_train_gpu, X_train_gpu.getDataPointer());
        Matrix::copyDataPointer(Y_train_cpu_, Y_train_cpu_.getDataPointer(0, 0, 0, net_id * Y_train_gpu.getNumber()), Y_train_gpu, Y_train_gpu.getDataPointer());
        //train_data_gpu.Y()->message("train data gpu Y");
        //发出拷贝数据结束信号
        train_info.data_distributed++;
        //调整学习率
        realc lr = net->adjustLearnRate(epoch_count);
        if (net_id == 0)
        {
            Log::LOG("Learn rate for epoch %d is %g\n", epoch_count, lr);
        }
        //训练前在训练集上的测试，若训练集实时生成可以使用
        if (net_id == 0 && epoch_count % test_epoch == 0 && pre_test_train)
        {
            net->test("Pre-test on train set", X_train_gpu, Y_train_gpu, A_train_gpu, 0, 1);
        }
        for (int iter = 0; iter < X_train_gpu.getNumber() / X_train_sub.getNumber(); iter++)
        {
            iter_count++;
            bool output = (iter + 1) % out_iter == 0;
            X_train_sub.shareData(X_train_gpu, 0, iter * X_train_sub.getNumber());
            Y_train_sub.shareData(Y_train_gpu, 0, iter * Y_train_sub.getNumber());
            //Log::LOG("%d, %g, %g\n", iter, train_data_sub.X()->dotSelf(), train_data_sub.Y()->dotSelf());

            //同步未完成
            WAIT_UNTIL(train_info.parameters_collected == 0);

            net->cal(true);
            //发出网络训练结束信号
            train_info.trained++;

            if (net_id == 0)
            {
                //主网络，完成信息输出，参数的收集和重新分发
                if (output)
                {
                    Log::LOG("epoch = %d, iter = %d\n", epoch_count, iter_count);
                }

                //主网络等待所有网络训练完成
                WAIT_UNTIL(train_info.trained == MP_count_);
                train_info.trained = 0;
                //同步
                if (MP_count_ > 1)
                {
                    for (int i = 1; i < nets.size(); i++)
                    {
                        Matrix::copyDataAcrossDevice(nets[i]->getCombinedWeight(), net->getWorkspaceWeight());
                        Matrix::add(net->getCombinedWeight(), net->getWorkspaceWeight(), net->getCombinedWeight());
                    }
                    net->getCombinedWeight().scale(1.0 / MP_count_);
                }
                //发布同步完成信号
                train_info.parameters_collected = MP_count_ - 1;
            }
            else
            {
                //非主网络等待同步结束
                WAIT_UNTIL(train_info.parameters_collected > 0);
                train_info.parameters_collected--;
                //分发到各个网络
                Matrix::copyDataAcrossDevice(nets[0]->getCombinedWeight(), net->getCombinedWeight());
            }
        }
        if (net_id == 0)
        {
            Log::LOG("Epoch %d finished\n", epoch_count);
        }
        //主网络负责测试
        if (net_id == 0 && epoch_count % test_epoch == 0)
        {
            std::string content = convert::formatString("Test net %d: ", net_id);
            realc test_result = 0;
            Matrix A;
            if (test_train_origin)
            {
                net->test(content + "original train set", X_train_, Y_train_, A, 0, test_max, 0);
            }
            if (test_train)
            {
                net->test(content + "transformed train set", X_train_gpu, Y_train_gpu, A, 0, test_max, 0);
            }
            if (dp_test_)
            {
                if (test_test_origin)
                {
                    net->test(content + "original test set", X_test_, Y_test_, A, 0, test_max, 0);
                    if (test_result >= max_test_origin_accuracy)
                    {
                        max_test_origin_accuracy = test_result;
                        max_test_origin_accuracy_epoch = epoch_count;
                    }
                }
                if (test_test)
                {
                    if (dp_test_->isFill())
                    {
                        Matrix::copyData(X_test_cpu_, X_test_gpu);
                        Matrix::copyData(Y_test_cpu_, Y_test_gpu);
                    }
                    net->test(content + "transformed test set", X_test_gpu, Y_test_gpu, A, 0, test_max, 0);
                    if (test_result >= max_test_accuracy)
                    {
                        max_test_accuracy = test_result;
                        max_test_accuracy_epoch = epoch_count;
                    }
                }
            }
            realc l1, l2;
            net->calNorm(l1, l2);
            Log::LOG("L1 = %g, L2 = %g\n", l1, l2);
            if (epoch_count % save_epoch == 0 && !save_format.empty())
            {
                std::string save_name = save_format;
                convert::replaceAllSubStringRef(save_name, "{epoch}", std::to_string(epoch_count));
                convert::replaceAllSubStringRef(save_name, "{date}", Timer::getNowAsString("%F"));
                convert::replaceAllSubStringRef(save_name, "{time}", Timer::getNowAsString("%T"));
                //convert::replaceAllSubStringRef(save_name, "{accurary}", std::to_string(test_accuracy_));
                convert::replaceAllSubStringRef(save_name, "\n", "");
                convert::replaceAllSubStringRef(save_name, " ", "_");
                convert::replaceAllSubStringRef(save_name, ":", "_");
                net->saveWeight(save_name);
            }
        }

        if (train_info.stop)
        {
            break;
        }
    }

    if (net_id == 0)
    {
        ConsoleControl::setColor(CONSOLE_COLOR_LIGHT_RED);
        if (test_test_origin)
        {
            Log::LOG("Maximum accuracy on original test set is %5.2f%% at epoch %d\n", max_test_origin_accuracy * 100, max_test_origin_accuracy_epoch);
        }
        if (test_test)
        {
            Log::LOG("Maximum accuracy on transformed test set is %5.2f%% at epoch %d\n", max_test_accuracy * 100, max_test_accuracy_epoch);
        }
        ConsoleControl::setColor(CONSOLE_COLOR_NONE);
    }
    fflush(nullptr);    //似乎偶然会发生最后一个权重文件写入失败，加上这个试试
}

//输出训练集和测试集的测试结果
void Neural::testData(Net* net, int force_output, int test_max)
{
    realc result;
    Matrix A(DeviceType::CPU);
    if (Option::getInstance().getInt("", "test_train"))
    {
        net->test("Test on train set", X_train_, Y_train_, A, force_output, test_max, 0);
    }
    if (Option::getInstance().getInt("", "test_train_cpu"))
    {
        net->test("Test on transformed train set", X_train_cpu_, Y_train_cpu_, A, force_output, test_max, 0);
    }
    if (Option::getInstance().getInt("", "test_test"))
    {
        net->test("Test on test set", X_test_, Y_test_, A, force_output, test_max, 0);
    }
    if (Option::getInstance().getInt("", "test_train_cpu"))
    {
        net->test("Test on transformed test set", X_test_cpu_, Y_test_cpu_, A, force_output, test_max, 0);
    }
}

//附加测试集，一般无用
void Neural::extraTest(Net* net, const std::string& section, int force_output, int test_max)
{
    if (!Option::getInstance().hasSection(section))
    {
        return;
    }
    auto dp_test = Factory::createDP(section, net->X().getDim(), net->Y().getDim());
    Matrix X, Y, A;
    dp_test->initData(X, Y);
    net->test("Extra test", X, Y, A, force_output, test_max);
}

int Neural::testExternalData(void* x, void* y, void* a, int n, int attack_times)
{
    Matrix X{ DeviceType::CPU }, Y{ DeviceType::CPU }, A{ DeviceType::CPU };
    auto net = getNet();

    auto dimx = net->X().getDim();
    dimx.back() = n;
    auto dimy = net->Y().getDim();
    dimy.back() = n;

    X.shareData((real*)x);
    Y.shareData((real*)y);
    A.shareData((real*)a);

    X.resize(dimx);
    Y.resize(dimy);
    A.resize(dimy);

    Log::setLog(0);

    return net->test("", X, Y, A, 0, 0, attack_times);
}

int Neural::getFloatPrecision()
{
    return sizeof(real);
}

}    // namespace woco