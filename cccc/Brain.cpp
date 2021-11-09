#include "Brain.h"
#include "ConsoleControl.h"
#include "File.h"
#include "NetCifa.h"
#include "NetLayer.h"
#include "Random.h"
#include "convert.h"
#include <algorithm>

namespace cccc
{

Brain::Brain() : X_train_{ DeviceType::CPU }, Y_train_{ DeviceType::CPU }, X_train_cpu_{ DeviceType::CPU }, Y_train_cpu_{ DeviceType::CPU }, X_test_{ DeviceType::CPU }, Y_test_{ DeviceType::CPU }, X_test_cpu_{ DeviceType::CPU }, Y_test_cpu_{ DeviceType::CPU }
{
}

Brain::~Brain()
{
    //delete_all(nets_);
    //cuda资源由操作系统回收
    //CudaControl::destroyAll();
}

//返回为0是正确创建
int Brain::init(const std::string& ini /*= ""*/)
{
    LOG("{}\n", Timer::getNowAsString());
    if (loadIni(ini))
    {
        LOG("Error: Load ini file failed!!\n");
        return 1;
    }
    if (testGPUDevice())
    {
        LOG("Error: GPU state is not right!!\n");
        return 2;
    }
    if (initNets())
    {
        LOG("Error: Initial net(s) failed!!\n");
        return 3;
    }
    if (initData())
    {
        LOG("Error: Initial data failed!!\n");
        return 4;
    }
    return 0;
}

int Brain::loadIni(const std::string& ini)
{
    //LOG("Size of real is %lu bytes\n", sizeof(real));

    //初始化选项
    //貌似这个设计比较瞎
    if (ini != "")
    {
        if (File::fileExist(ini))
        {
            option_.loadFile(ini);
        }
        else
        {
            option_.loadString(ini);
        }
    }
    //option_.print();
    batch_ = (std::max)(1, option_.getInt("train", "batch", 100));
    //work_mode_ = option_.getEnum("", "work_mode", WORK_MODE_NORMAL);
    LOG::setLevel(option_.getInt("train", "output_log", 1));
    return 0;
}

int Brain::testGPUDevice()
{
    //gpu测试
    int device_count = 0;
    if (option_.getInt("train", "use_cuda", 1))
    {
        device_count = CudaControl::checkDevices();
        if (device_count > 0)
        {
            LOG("Found {} CUDA device(s)\n", device_count);
            CudaControl::setGlobalCudaType(DeviceType::GPU);
        }
        else
        {
            LOG("Error: No CUDA devices!!\n");
            CudaControl::setGlobalCudaType(DeviceType::CPU);
        }
    }

    if (option_.getInt("train", "use_cuda") != 0 && CudaControl::getGlobalCudaType() != DeviceType::GPU)
    {
        LOG("CUDA state is not right, refuse to run!\n");
        LOG("Re-init the net again, or consider CPU mode (manually set use_cuda = 0).\n");
        return 1;
    }

    MP_count_ = 1;
    if (MP_count_ <= 0)
    {
        MP_count_ = 1;
    }
    return 0;
}

int Brain::initNets()
{
    nets_.clear();
    nets_.resize(MP_count_);
    //这里读取ini中指定的device顺序，其中第一个设备为主网络，该值一般来说应由用户指定
    auto mp_device = option_.getVector<int>("train", "mp_device");
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
    auto cifa = option_.getInt("train", "cifa");
    for (int i = 0; i < MP_count_; i++)
    {
        LOG("Trying to create net {}...\n", i);
        int cuda_id = -1;
        if (CudaControl::select(mp_device[i]))
        {
            cuda_id = CudaControl::select(mp_device[i])->getDeviceID();    //若不合理则换成一个能用的
        }
        if (mp_device[i] != cuda_id)
        {
            LOG("Device {} is not right, change it to {}\n", mp_device[i], cuda_id);
            mp_device[i] = cuda_id;
        }
        std::unique_ptr<Net> net;
        if (cifa)
        {
            net = std::make_unique<NetCifa>();
        }
        else
        {
            net = std::make_unique<NetLayer>();
        }
        net->setDeviceID(mp_device[i]);
        int dev_id = net->getDeviceID();
        net->setOption(&option_);
        if (net->init() != 0)
        {
            return 1;
        }
        if (option_.getInt("train", "load_net") != 0)
        {
            net->loadWeight(option_.getString("train", "load_file"));
        }
        nets_[i] = std::move(net);
        if (dev_id >= 0)
        {
            LOG("Net {} has been created on device {}\n", i, dev_id);
        }
        else
        {
            LOG("Net {} has been created on CPU\n", i);
        }
    }

    if (MP_count_ > 1)
    {
        for (int i = 0; i < nets_.size(); i++)
        {
            auto& net = nets_[i];
            //只使用0号网络的权值
            if (i != 0)
            {
                Matrix::copyDataAcrossDevice(nets_[0]->getAllWeights(), net->getAllWeights());
            }
        }
    }

    //主线程使用0号网络
    nets_[0]->setDeviceSelf();
    return 0;
}

int Brain::initData()
{
    initDataPreparer();
    initTrainData();
    initTestData();
    return 0;
}

void Brain::initDataPreparer()
{
    //数据准备器
    //这里使用公共数据准备器，实际上完全可以创建私有的准备器
    auto dim0 = nets_[0]->getX().getDim();
    auto dim1 = nets_[0]->getY().getDim();
    //DataPreparerFactory::destroy(data_preparer_);
    //data_preparer_ =  std::make_unique<>  DataPreparerFactory::create(&option_, "data_preparer", dim0, dim1);
    data_preparer_ = DataPreparerFactory::makeUniquePtr(&option_, "data_preparer", dim0, dim1);
    std::string test_section = "data_preparer2";
    if (!option_.hasSection(test_section))
    {
        option_.setKey("", "test_test", "0");
        option_.setKey("", "test_test_origin", "0");
        return;
        //option_.setOption(test_section, "test", "1");
    }
    data_preparer2_ = DataPreparerFactory::makeUniquePtr(&option_, test_section, dim0, dim1);
}

//初始化训练集，必须在DataPreparer之后
void Brain::initTrainData()
{
    data_preparer_->initData(X_train_, Y_train_);
    //data_preparer_->resizeDataGroup(train_data_origin_);

    //训练数据使用的显存量，不能写得太小
    double size_gpu = option_.getReal("train", "cuda_max_train_space", 1e9);
    //计算显存可以放多少组数据，为了方便计算，要求是minibatch的整数倍，且可整除原始数据组数
    data_preparer_->prepareData(0, "train", X_train_, Y_train_, X_train_cpu_, Y_train_cpu_);
}

//生成测试集
void Brain::initTestData()
{
    if (data_preparer2_)
    {
        data_preparer2_->initData(X_test_, Y_test_);
        data_preparer2_->prepareData(0, "test", X_test_, Y_test_, X_test_cpu_, Y_test_cpu_);
    }
}

//运行，注意容错保护较弱
//注意通常情况下是使用第一个网络测试数据
void Brain::run(int train_epochs /*= -1*/)
{
    auto& net = nets_[0];
    //初测
    testData(net.get(), option_.getInt("train", "force_output"), option_.getInt("train", "test_type", 1));

    if (train_epochs < 0)
    {
        train_epochs = option_.getInt("train", "train_epochs", 20);
    }
    LOG("Going to run for {} epochs...\n", train_epochs);

    train(nets_, data_preparer_.get(), train_epochs);

    std::string save_filename = option_.getString("train", "save_file");
    if (save_filename != "")
    {
        net->saveWeight(save_filename);
    }

    //终测
    testData(net.get(), option_.getInt("train", "force_output"), option_.getInt("train", "test_type", 1));
    //附加测试，有多少个都能用
    extraTest(net.get(), "extra_test", option_.getInt("train", "force_output"), option_.getInt("train", "test_type", 1));

#ifdef LAYER_TIME
    for (auto& l : nets_[0]->getLayerVector())
    {
        LOG("{}: {},{},{}\n", l->getName(), l->total_time1_, l->total_time2_, l->total_time3_);
    }
#endif

    auto time_sec = timer_total_.getElapsedTime();
    LOG("Run neural net end. Total time is {}\n", Timer::autoFormatTime(time_sec));
    LOG("{}\n", Timer::getNowAsString());
}

//暂时阻塞等待条件。注意，这里只能用宏，用函数写起来很麻烦
#define WAIT_UNTIL(condition) \
    do { \
        while (!(condition)) { std::this_thread::sleep_for(std::chrono::nanoseconds(100)); } \
    } while (0)
#define WAIT_UNTIL_OVERTIME(condition, overtime) \
    do { \
        auto t0 = std::chrono::system_clock::now(); \
        while (!(condition) && std::chrono::system_clock::now() - t0 < std::chrono::nanoseconds(int64_t(overtime))) \
        { \
            std::this_thread::sleep_for(std::chrono::nanoseconds(100)); \
        } \
    } while (0)

//训练一批数据，输出步数和误差，若训练次数为0可以理解为纯测试模式
//首个参数为指定几个结构完全相同的网络并行训练
void Brain::train(std::vector<std::unique_ptr<Net>>& nets, DataPreparer* data_preparer, int epochs)
{
    if (epochs <= 0)
    {
        return;
    }

    int iter_per_epoch = X_train_.getNumber() / batch_ / MP_count_;    //如果不能整除，则估计会不准确，但是关系不大
    if (iter_per_epoch <= 0)
    {
        iter_per_epoch = 1;
    }
    epoch_count_ = iter_count_ / iter_per_epoch;

    real e = 0, e0 = 0;

    Timer timer_per_epoch, timer_trained;
    //prepareData();
    TrainInfo train_info;
    train_info.data_prepared = -1;

    int test_test = option_.getInt("train", "test_test", 0);
    int test_test_origin = option_.getInt("train", "test_test_origin", 0);
    int test_epoch = option_.getInt("train", "test_epoch", 1);

    //创建训练进程
    std::vector<std::thread> net_threads(nets.size());
    for (int i = 0; i < net_threads.size(); i++)
    {
        net_threads[i] = std::thread{ [this, &nets, i, &train_info, epochs]()
            { trainOneNet(nets, i, train_info, epoch_count_, epochs); } };
    }

    train_info.stop = 0;
    int epoch0 = epoch_count_;
    for (int epoch_count = epoch0; epoch_count < epoch0 + epochs; epoch_count++)
    {
        data_preparer->prepareData(epoch_count, "train", X_train_, Y_train_, X_train_cpu_, Y_train_cpu_);
        if (data_preparer2_ && (test_test || test_test_origin) && epoch_count % test_epoch == 0 && data_preparer2_->isFill())
        {
            data_preparer2_->prepareData(epoch_count, "test", X_test_, Y_test_, X_test_cpu_, Y_test_cpu_);
        }
        train_info.data_prepared = epoch_count;
        WAIT_UNTIL(train_info.data_distributed == MP_count_ || train_info.stop);
        //train_info.data_prepared = 0;
        train_info.data_distributed = 0;
        //回调
        if (running_callback_)
        {
            running_callback_(this);
        }
        iter_count_ += iter_per_epoch;
        epoch_count_++;
        if (epoch_count - epoch0 > 0)
        {
            double left_time = timer_trained.getElapsedTime() / (epoch_count - epoch0) * (epoch0 + epochs - epoch_count);    //注意这个估测未考虑额外测试的消耗，不是很准
            LOG("{} s for this epoch, {} elapsed totally, about {} left\n",
                timer_per_epoch.getElapsedTime(), Timer::autoFormatTime(timer_total_.getElapsedTime()), Timer::autoFormatTime(left_time));
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

    LOG("{} s for this epoch, {} elapsed totally\n", timer_per_epoch.getElapsedTime(), Timer::autoFormatTime(timer_total_.getElapsedTime()));
    LOG("\n");
}

//训练网络数组nets中的一个
void Brain::trainOneNet(std::vector<std::unique_ptr<Net>>& nets, int net_id, TrainInfo& train_info, int epoch0, int epochs)
{
    auto& net = nets[net_id];
    net->setDeviceSelf();
    net->setActivePhase(ACTIVE_PHASE_TRAIN);

    auto x_dim = X_train_cpu_.getDim();
    auto y_dim = Y_train_cpu_.getDim();
    x_dim.back() /= MP_count_;
    y_dim.back() /= MP_count_;
    Matrix X_train_gpu(x_dim), Y_train_gpu(y_dim), A_train_gpu(y_dim);
    Matrix X_train_sub(net->getX()), Y_train_sub(net->getY());
    //LOG("%g, %g\n", train_data_origin_.X()->dotSelf(), train_data_cpu_.Y()->dotSelf());

    //Matrix X_test_data_gpu(MATRIX_DATA_INSIDE, DeviceType::GPU);
    //if (net_id == 0)
    //{
    //test_data_gpu.copyFrom(test_data_cpu_);
    //}

    int test_train = option_.getInt("train", "test_train", 0);
    int test_train_origin = option_.getInt("train", "test_train_origin", 0);
    int pre_test_train = option_.getInt("train", "pre_test_train", 0);
    int test_test = option_.getInt("train", "test_test", 0);
    int test_test_origin = option_.getInt("train", "test_test_origin", 0);
    int test_epoch = option_.getInt("train", "test_epoch", 1);
    int save_epoch = option_.getInt("train", "save_epoch", 10);
    int out_iter = option_.getInt("train", "out_iter", 100);
    int test_type = option_.getInt("train", "test_type", 1);
    std::string save_format = option_.dealString(option_.getString("train", "save_format", "save/save-{epoch}.txt"), 1);
    int total_batch = X_train_cpu_.getNumber();

    realc max_test_origin_accuracy = 0;
    int max_test_origin_accuracy_epoch = 0;
    realc max_test_accuracy = 0;
    int max_test_accuracy_epoch = 0;

    int iter_count = 0;
    int epoch_count = epoch0;
    while (epoch_count < epoch0 + epochs)
    {
        //等待数据准备完成
        WAIT_UNTIL(train_info.data_prepared == epoch_count || train_info.stop);
        epoch_count++;
        Matrix::copyRows(X_train_cpu_, net_id * total_batch / MP_count_, X_train_gpu, 0, X_train_gpu.getNumber());
        Matrix::copyRows(Y_train_cpu_, net_id * total_batch / MP_count_, Y_train_gpu, 0, Y_train_gpu.getNumber());
        //train_data_gpu.Y()->message("train data gpu Y");
        //发出拷贝数据结束信号
        train_info.data_distributed++;
        //调整学习率
        realc lr = net->adjustLearnRate(epoch_count);
        if (net_id == 0)
        {
            LOG("Learn rate for epoch {} is {}\n", epoch_count, lr);
        }
        //训练前在训练集上的测试，若训练集实时生成可以使用
        if (net_id == 0 && epoch_count % test_epoch == 0 && pre_test_train)
        {
            net->test("Test on train set before training", &X_train_gpu, &Y_train_gpu, &A_train_gpu, 0, 1);
            net->checkNorm();
        }
        realc e = 0;
        for (int iter = 0; iter < X_train_gpu.getNumber() / X_train_sub.getNumber(); iter++)
        {
            iter_count++;
            bool output = (iter + 1) % out_iter == 0;
            X_train_sub.shareData(X_train_gpu, 0, iter * X_train_sub.getNumber());
            Y_train_sub.shareData(Y_train_gpu, 0, iter * Y_train_sub.getNumber());
            //LOG("%d, %g, %g\n", iter, train_data_sub.X()->dotSelf(), train_data_sub.Y()->dotSelf());

            //同步未完成
            WAIT_UNTIL(train_info.parameters_collected == 0 || train_info.stop);

            net->active(&X_train_sub, &Y_train_sub, nullptr, true, output ? &e : nullptr);
            //发出网络训练结束信号
            train_info.trained++;

            if (net_id == 0)
            {
                //主网络，完成信息输出，参数的收集和重新分发
                if (output)
                {
                    LOG("epoch = {}, iter = {}, error = {:.8}\n", epoch_count, iter_count, e);
                }

                //主网络等待所有网络训练完成
                WAIT_UNTIL(train_info.trained == MP_count_ || train_info.stop);
                train_info.trained = 0;
                //同步
                if (MP_count_ > 1)
                {
                    for (int i = 1; i < nets.size(); i++)
                    {
                        Matrix::copyDataAcrossDevice(nets[i]->getAllWeights(), net->getWorkspace());
                        Matrix::add(net->getAllWeights(), net->getWorkspace(), net->getAllWeights());
                    }
                    net->getAllWeights().scale(1.0 / MP_count_);
                }
                //发布同步完成信号
                train_info.parameters_collected = MP_count_ - 1;
            }
            else
            {
                //非主网络等待同步结束
                WAIT_UNTIL(train_info.parameters_collected > 0 || train_info.stop);
                train_info.parameters_collected--;
                //分发到各个网络
                Matrix::copyDataAcrossDevice(nets[0]->getAllWeights(), net->getAllWeights());
            }
        }
        if (net_id == 0)
        {
            LOG("Epoch {} finished\n", epoch_count);
        }
        //主网络负责测试
        if (net_id == 0 && epoch_count % test_epoch == 0)
        {
            std::string content = format1::format("Test net {}: ", net_id);
            realc test_result;
            if (test_train_origin)
            {
                net->test(content + "original train set", &X_train_, &Y_train_, nullptr, 0, test_type, 0, &test_result);
            }
            if (test_train)
            {
                net->test(content + "transformed train set", &X_train_gpu, &Y_train_gpu, nullptr, 0, test_type, 0, &test_result);
            }
            if (data_preparer2_)
            {
                if (test_test_origin)
                {
                    net->test(content + "original test set", &X_test_, &Y_test_, nullptr, 0, test_type, 0, &test_result);
                    if (test_result >= max_test_origin_accuracy)
                    {
                        max_test_origin_accuracy = test_result;
                        max_test_origin_accuracy_epoch = epoch_count;
                    }
                }
                if (test_test)
                {
                    net->test(content + "transformed test set", &X_test_cpu_, &Y_test_cpu_, nullptr, 0, test_type, 0, &test_result);
                    if (test_result >= max_test_accuracy)
                    {
                        max_test_accuracy = test_result;
                        max_test_accuracy_epoch = epoch_count;
                    }
                }
            }
            if (net->checkNorm())
            {
                LOG("Numerical fault! Stop training!\n");
                train_info.stop = 1;
            }
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
            LOG("Maximum accuracy on original test set is {:5.2f}% at epoch {}\n", max_test_origin_accuracy * 100, max_test_origin_accuracy_epoch);
        }
        if (test_test)
        {
            LOG("Maximum accuracy on transformed test set is {:5.2f}% at epoch {}\n", max_test_accuracy * 100, max_test_accuracy_epoch);
        }
        ConsoleControl::setColor(CONSOLE_COLOR_NONE);
    }
}

//输出训练集和测试集的测试结果
void Brain::testData(Net* net, int force_output /*= 0*/, int test_type /*= 0*/)
{
    if (net == nullptr)
    {
        net = nets_[0].get();
    }
    realc result;
    if (option_.getInt("train", "test_train_origin"))
    {
        net->test("Test on original train set", &X_train_, &Y_train_, nullptr, force_output, test_type, 0, &result);
    }
    if (option_.getInt("train", "test_train"))
    {
        net->test("Test on transformed train set", &X_train_cpu_, &Y_train_cpu_, nullptr, force_output, test_type, 0, &result);
    }
    if (option_.getInt("train", "test_test_origin"))
    {
        net->test("Test on original test set", &X_test_, &Y_test_, nullptr, force_output, test_type, 0, &result);
    }
    if (option_.getInt("train", "test_test"))
    {
        net->test("Test on transformed test set", &X_test_cpu_, &Y_test_cpu_, nullptr, force_output, test_type, 0, &result);
    }
    net->checkNorm();
}

//附加测试集，一般无用
void Brain::extraTest(Net* net, const std::string& section, int force_output /*= 0*/, int test_type /*= 0*/)
{
    if (!option_.hasSection(section))
    {
        return;
    }
    auto dp_test = DataPreparerFactory::makeUniquePtr(&option_, section, net->getX().getDim(), net->getY().getDim());
    Matrix X_extra, Y_extera;
    dp_test->initData(X_extra, Y_extera);
    if (X_extra.getDataSize() > 0 && Y_extera.getDataSize() > 0)
    {
        net->test("Extra test", &X_extra, &Y_extera, nullptr, force_output, test_type);
    }
}

int Brain::testExternalData(void* x, void* y, void* a, int n, int attack_times, realc* error)
{
    std::vector<int> n_begin(1, 0), n_count(1, n);
    auto run = [this, x, y, a, attack_times, error, &n_begin, &n_count](int i)
    {
        Matrix X(DeviceType::CPU), Y(DeviceType::CPU), A(DeviceType::CPU);

        auto net = nets_[i].get();
        auto dim0 = net->getX().getDim();
        dim0.back() = n_count[i];
        auto dim1 = net->getY().getDim();
        dim1.back() = n_count[i];

        Matrix *x1 = nullptr, *y1 = nullptr, *a1 = nullptr;
        if (x)
        {
            X.shareData((real*)x + net->getX().getRow() * n_begin[i]);
            X.resize(dim0);
            x1 = &X;
        }
        if (y)
        {
            Y.shareData((real*)y + net->getY().getRow() * n_begin[i]);
            Y.resize(dim1);
            y1 = &Y;
        }
        if (a)
        {
            A.shareData((real*)a + net->getA().getRow() * n_begin[i]);
            A.resize(dim1);
            a1 = &A;
        }

        return net->test("", x1, y1, a1, 0, 0, attack_times, error);
    };
    LOG::setLevel(0);
    if (nets_.size() == 1)
    {
        run(0);
    }
    else if (nets_.size() > 1)
    {
        n_begin.resize(nets_.size());
        n_count.resize(nets_.size());
        int k = (n + nets_.size() - 1) / nets_.size();
        for (int i = 0; i < nets_.size(); i++)
        {
            n_begin[i] = k * i;
            n_count[i] = std::min(n - k * i, k);
        }
#pragma omp parallel for
        for (int i = 0; i < nets_.size(); i++)
        {
            run(i);
        }
    }
    LOG::restoreLevel();
    return 0;
}

}    // namespace cccc