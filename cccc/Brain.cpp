#include "Brain.h"
#include "ConsoleControl.h"
#include "DynamicLibrary.h"
#include "NetCifa.h"
#include "NetLayer.h"
#include "filefunc.h"
#include "strfunc.h"
#include <algorithm>
#include <cmath>

namespace cccc
{

Brain::Brain()
{
}

Brain::~Brain()
{
}

//返回为0是正确创建
int Brain::init(const std::string& ini /*= ""*/)
{
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
    LOG("{}\n", Timer::getNowAsString());
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
    //LOG("Size of real is {}\n", sizeof(real));

    //初始化选项
    //貌似这个设计比较瞎
    if (ini != "")
    {
        if (filefunc::fileExist(ini))
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
    //work_mode_ = option_.getEnum("train", "work_mode", WORK_MODE_NORMAL);
    return 0;
}

int Brain::testGPUDevice()
{
    //gpu测试
    int device_count = 0;
    if (option_.getInt("train", "gpu", 1))
    {
        GpuControl::checkDevices();
        device_count = GpuControl::getDeviceCount();
    }

    if (option_.getInt("train", "gpu") != 0 && device_count == 0)
    {
        LOG("GPU state is not right, refuse to run!\n");
        LOG("Reinstall nvidia drivers and libraries again, or consider CPU mode (manually set gpu = 0).\n");
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
        auto cude_device_turn = GpuControl::cudaDevicesTurn();
        if (cude_device_turn.empty())
        {
            mp_device[0] = 0;
        }
        else
        {
            for (int i = 0; i < MP_count_; i++)
            {
                mp_device[i] = cude_device_turn[i];
            }
        }
    }
    gpus_.resize(mp_device.size());
    for (int i = 0; i < mp_device.size(); i++)
    {
        gpus_[i].init(mp_device[i]);
        //检查显存是否足够，注意此处实际上是初始化后的，故可能已经有一部分显存已被实例占用
        size_t free_mem, total_mem;
        gpus_[i].getFreeMemory(free_mem, total_mem);
        double need_free = option_.getReal("train", "need_free_mem", 0.6);
        if (1.0 * free_mem / total_mem < need_free)
        {
            LOG("Error: Not enough memory on device {}!!\n", mp_device[i]);
            LOG("Need {}%, but only {}% free\n", need_free * 100, 100.0 * free_mem / total_mem);
            return 1;
        }
    }
    auto cifa = option_.getInt("net", "cifa", 1);
    for (int i = 0; i < MP_count_; i++)
    {
        LOG("Trying to create net {}...\n", i);
        gpus_[i].setAsCurrent();
        std::unique_ptr<Net> net;
        if (0)
        {
        }
        else
        {
            if (cifa)
            {
                net = std::make_unique<NetCifa>();
            }
            else
            {
                net = std::make_unique<NetLayer>();
            }
        }
        net->setGpu(&gpus_[i]);
        int dev_id = net->getCuda()->getDeviceID();
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
    if (option_.hasSection("data_preparer"))
    {
        data_preparer_ = DataPreparerFactory::makeUniquePtr(&option_, "data_preparer", dim0, dim1);
    }
    if (option_.hasSection("data_preparer2"))
    {
        data_preparer2_ = DataPreparerFactory::makeUniquePtr(&option_, "data_preparer2", dim0, dim1);
    }
    else
    {
        option_.setKey("train", "test_test", "0");
        option_.setKey("train", "test_test_origin", "0");
    }
}

//初始化训练集，必须在DataPreparer之后
void Brain::initTrainData()
{
    if (data_preparer_)
    {
        data_preparer_->prepareData(0, "train");
    }
}

//生成测试集
void Brain::initTestData()
{
    if (data_preparer2_)
    {
        data_preparer2_->prepareData(0, "test");
    }
}

std::string Brain::makeSaveSign()
{
    std::string sign = option_.getString("train", "save_sign");
    sign = fmt1::format("{}, {}, {}, {}",
        option_.dealString(sign, true),
        Timer::getNowAsString(),
        Timer::autoFormatTime(timer_total_.getElapsedTime()),
        data_preparer_->X.getNumber() * epoch_count_);
    return sign;
}

//运行，注意容错保护较弱
//注意通常情况下是使用第一个网络测试数据
void Brain::run(int train_epochs /*= -1*/)
{
    auto& net = nets_[0];

    //初测
    //使用CPU时不测试，终测同
    if (GpuControl::getGlobalCudaType() == UnitType::GPU)
    {
        testData(net.get(), option_.getInt("train", "force_output"), option_.getInt("train", "test_type", 1));
    }
    if (train_epochs < 0)
    {
        train_epochs = option_.getInt("train", "train_epochs", 20);
    }

    train(nets_, data_preparer_.get(), train_epochs);

    std::string save_filename = option_.getString("train", "save_file");
    if (save_filename != "")
    {
        net->saveWeight(save_filename, makeSaveSign());
    }
    net->getSolver().outputState();
    //终测
    if (GpuControl::getGlobalCudaType() == UnitType::GPU)
    {
        testData(net.get(), option_.getInt("train", "force_output"), option_.getInt("train", "test_type", 1));
    }
    //附加测试，有多少个都能用
    extraTest(net.get(), "extra_test", option_.getInt("train", "force_output"), option_.getInt("train", "test_type", 1));

#ifdef LAYER_TIME
    for (auto& l : nets_[0]->getLayerVector())
    {
        LOG("{}: {},{},{}\n", l->getName(), l->total_time1_, l->total_time2_, l->total_time3_);
    }
#endif

    auto time_sec = timer_total_.getElapsedTime();
    LOG("Total time is {}\n", Timer::autoFormatTime(time_sec));
    LOG("End at {}\n", Timer::getNowAsString());
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

//训练一批数据，输出步数和误差
//首个参数为指定几个结构完全相同的网络并行训练
void Brain::train(std::vector<std::unique_ptr<Net>>& nets, DataPreparer* data_preparer, int total_epochs)
{
    if (total_epochs <= 0)
    {
        return;
    }

    int iter_per_epoch = data_preparer_->X.getNumber() / batch_ / MP_count_;    //如果不能整除，则估计会不准确，但是关系不大
    if (iter_per_epoch <= 0)
    {
        iter_per_epoch = 1;
    }
    iter_count_ = 0;
    epoch_count_ = 0;

    real e = 0, e0 = 0;
    int test_test = option_.getInt("train", "test_test", 0);
    int test_test_origin = option_.getInt("train", "test_test_origin", 0);
    int test_epoch = option_.getInt("train", "test_epoch", 1);

    TrainInfo train_info;
    train_info.data_prepared = -1;
    //创建训练进程
    std::vector<std::thread> net_threads(nets.size());
    for (int i = 0; i < net_threads.size(); i++)
    {
        net_threads[i] = std::thread{ [this, &nets, i, &train_info, total_epochs]()
            {
                trainOneNet(nets, i, train_info, total_epochs);
            } };
    }

    train_info.stop = 0;

    Timer timer_per_epoch;    //计时

    for (int epoch_count = 0;; epoch_count++)
    {
        data_preparer->prepareData(epoch_count, "train");
        if (data_preparer2_ && (test_test || test_test_origin) && epoch_count % test_epoch == test_epoch - 1 && data_preparer2_->isFill())
        {
            data_preparer2_->prepareData(epoch_count, "test");
        }
        train_info.data_prepared = epoch_count;
        WAIT_UNTIL(train_info.data_distributed == MP_count_ || train_info.stop);
        //train_info.data_prepared = 0;
        train_info.data_distributed = 0;
        //回调
        if (running_callback_) { running_callback_(this); }
        iter_count_ += iter_per_epoch;
        epoch_count_++;
        if (epoch_count > 0)
        {
            LOG("{} s for this epoch, {} elapsed totally\n",
                timer_per_epoch.getElapsedTime(), Timer::autoFormatTime(timer_total_.getElapsedTime()));
        }
        timer_per_epoch.start();
        if (train_info.stop) { break; }
    }

    train_info.stop = 1;

    for (int i = 0; i < net_threads.size(); i++)
    {
        net_threads[i].join();
    }

    LOG("{} s for this epoch, {} elapsed totally\n", timer_per_epoch.getElapsedTime(), Timer::autoFormatTime(timer_total_.getElapsedTime()));
    LOG("\n");
}

//训练网络数组nets中的一个
//在并行时需注意batch的设置，可以认为batch（非并行） = batch（并行）*线程数
void Brain::trainOneNet(std::vector<std::unique_ptr<Net>>& nets, int net_id, TrainInfo& train_info, int total_epochs)
{
    auto& net = nets[net_id];
    net->setDeviceSelf();
    net->setActivePhase(ACTIVE_PHASE_TRAIN);

    auto x_dim = data_preparer_->X.getDim();
    auto y_dim = data_preparer_->Y.getDim();
    x_dim.back() /= MP_count_;
    y_dim.back() /= MP_count_;
    Matrix X_train_gpu(x_dim), Y_train_gpu(y_dim), A_train_gpu(y_dim), lw_train_gpu;

    if (data_preparer_->LW.getDataSize() > 0)
    {
        lw_train_gpu.resize(y_dim);
        net->initLossWeight();
    }

    int test_epoch = option_.getInt("train", "test_epoch", 1);
    int test_train = option_.getInt("train", "test_train", 0);
    int test_train_origin = option_.getInt("train", "test_train_origin", 0);
    int pre_test_train = option_.getInt("train", "pre_test_train", 0);
    int test_test = option_.getInt("train", "test_test", 0);
    int test_test_origin = option_.getInt("train", "test_test_origin", 0);

    int save_epoch = option_.getInt("train", "save_epoch", 10);
    int out_iter = option_.getInt("train", "out_iter", 100);
    int test_type = option_.getInt("train", "test_type", 1);

    std::string save_format = option_.dealString(option_.getString("train", "save_format", "save/save-{epoch}.txt"), false);
    int total_batch = data_preparer_->X.getNumber();

    realc max_test_origin_accuracy = 0;
    int max_test_origin_accuracy_epoch = 0;
    realc max_test_accuracy = 0;
    int max_test_accuracy_epoch = 0;

    int iter_count = 0;
    int epoch_count = 0;
    int effective_epoch_count = 0;    //有效epoch计数，注意仅主网络的这个值在更新
    //int effective_epoch_count_saved = 0;

    Matrix weight_backup(net->getAllWeights().getDim());    //备份权重
    Matrix workspace;
    Matrix::copyData(net->getAllWeights(), weight_backup);
    Matrix::copyData(net->getAllWeights().d(), weight_backup.d());

    double time_limited = option_.getReal("train", "time_limited", 86400 * 100);

    Timer timer_test, timer_trained, t_out;
    double time_test0 = 0;
    double time_per_epoch = 1;

    realc l1p = 0, l2p = 0;
    //while (epoch_count < epochs)
    while (train_info.stop == 0)
    {
        //等待数据准备完成
        WAIT_UNTIL(train_info.data_prepared == epoch_count || train_info.stop);
        epoch_count++;
        Matrix::copyRows(data_preparer_->X, net_id * total_batch / MP_count_, X_train_gpu, 0, X_train_gpu.getNumber());
        Matrix::copyRows(data_preparer_->Y, net_id * total_batch / MP_count_, Y_train_gpu, 0, Y_train_gpu.getNumber());
        Matrix::copyRows(data_preparer_->LW, net_id * total_batch / MP_count_, lw_train_gpu, 0, lw_train_gpu.getNumber());
        //train_data_gpu.Y()->message("train data gpu Y");
        //发出拷贝数据结束信号
        train_info.data_distributed++;
        //调整学习率
        //按时间算的奇怪参数
        int progress_by_time = timer_trained.getElapsedTime() / time_limited * total_epochs;
        if (progress_by_time < epoch_count)
        {
            net->adjustByEpoch(epoch_count, total_epochs);
        }
        else
        {
            net->adjustByEpoch(progress_by_time, total_epochs);
        }

        if (net_id == 0)
        {
            LOG("Learn rate for epoch {} is {}\n", epoch_count, net->getSolver().getLearnRate());
        }
        //训练前在训练集上的测试，若训练集实时生成可以使用
        if (net_id == 0 && epoch_count % test_epoch == 0 && pre_test_train)
        {
            net->test("Test on train set before training", &X_train_gpu, &Y_train_gpu, &A_train_gpu, 0, 1);
            net->outputNorm();
        }
        realc e = 0;
        for (int iter = 0; iter < X_train_gpu.getNumber() / net->getX().getNumber(); iter++)
        {
            iter_count++;
            bool output = (iter + 1) % out_iter == 0;
            net->getX().shareData(X_train_gpu, 0, iter * net->getX().getNumber());
            net->getY().shareData(Y_train_gpu, 0, iter * net->getY().getNumber());
            net->getLossWeight().shareData(lw_train_gpu, 0, iter * net->getLossWeight().getNumber());

            //同步未完成
            WAIT_UNTIL(train_info.parameters_collected == 0 || train_info.stop);
            //Timer t;
            net->active(nullptr, nullptr, nullptr, true, output ? &e : nullptr);
            //net->outputNorm();
            //LOG("Iter {} finished in {} , {}s\n", iter_count, t.getElapsedTime(), t_out.getElapsedTime());
            t_out.start();
            //发出网络反向结束信号
            train_info.trained++;

            if (net_id == 0)
            {
                //LOG("{}\n", t_out.getElapsedTime());
                //t_out.start();
                //主网络，完成信息输出，参数的收集和重新分发
                if (output)
                {
                    LOG("epoch = {}, iter = {}, error = {:.8}\n", epoch_count, iter_count, e);
                }
                //主网络等待所有网络反向完成
                WAIT_UNTIL(train_info.trained == MP_count_ || train_info.stop);
                train_info.trained = 0;
                //同步
                if (MP_count_ > 1)
                {
                    auto calSum = [&workspace, &net, &nets, this](auto func)
                    {
                        workspace.resize(func(net));
                        for (int i = 1; i < nets.size(); i++)
                        {
                            Matrix::copyDataAcrossDevice(func(nets[i]), workspace);
                            Matrix::add(func(net), workspace, func(net));
                        }
                    };
                    calSum([](std::unique_ptr<Net>& n) -> Matrix&
                        {
                            return n->getAllWeights().d();
                        });
                }
                //主网络更新权重
                net->updateWeight();
                //梯度在数学上应是直接求和，但因后续经常保留一部分梯度，求和会导致数值越来越大
                net->getAllWeights().d().scale(1.0 / MP_count_);
                //主网络负责测试，因可能出现回退的情况，故需在分发前处理
                if (iter == X_train_gpu.getNumber() / net->getX().getNumber() - 1)
                {
                    //LOG("Epoch {} finished\n", epoch_count);
                    if (epoch_count % test_epoch == 0)
                    {
                        std::string content = fmt1::format("Test net {}: ", net_id);
                        realc test_result;
                        int test_error = 0;
                        if (test_train_origin)
                        {
                            test_error += net->test(content + "original train set", &data_preparer_->X, &data_preparer_->Y, nullptr, 0, test_type, 0, &test_result);
                        }
                        if (test_train)
                        {
                            test_error += net->test(content + "transformed train set", &X_train_gpu, &Y_train_gpu, nullptr, 0, test_type, 0, &test_result);
                        }
                        if (data_preparer2_)
                        {
                            if (test_test_origin)
                            {
                                test_error += net->test(content + "original test set", &data_preparer2_->X0, &data_preparer2_->Y0, nullptr, 0, test_type, 0, &test_result);
                                if (test_result >= max_test_origin_accuracy)
                                {
                                    max_test_origin_accuracy = test_result;
                                    max_test_origin_accuracy_epoch = epoch_count;
                                }
                            }
                            if (test_test)
                            {
                                test_error += net->test(content + "transformed test set", &data_preparer2_->X, &data_preparer2_->Y, nullptr, 0, test_type, 0, &test_result);
                                if (test_result >= max_test_accuracy)
                                {
                                    max_test_accuracy = test_result;
                                    max_test_accuracy_epoch = epoch_count;
                                }
                            }
                        }
                        Matrix::copyData(net->getAllWeights(), weight_backup);
                        Matrix::copyData(net->getAllWeights().d(), weight_backup.d());
                        auto time_test = timer_test.getElapsedTime();
                        time_per_epoch = (time_test - time_test0) / test_epoch;
                        time_test0 = time_test;
                        double time_rest;
                        time_rest = (total_epochs - epoch_count) * time_per_epoch;
                        time_rest = std::min(time_rest, time_limited - timer_trained.getElapsedTime());
                        time_rest = std::max(time_rest, 0.0);
                        net->getSolver().outputState();
                        LOG("About {} left\n", Timer::autoFormatTime(time_rest));
                    }
                }
                //发布收集完成信号
                train_info.parameters_collected = MP_count_ - 1;
            }
            else
            {
                //非主网络等待收集结束
                WAIT_UNTIL(train_info.parameters_collected > 0 || train_info.stop);
                train_info.parameters_collected--;
                //更新到自己的权重
                Matrix::copyDataAcrossDevice(nets[0]->getAllWeights(), net->getAllWeights());
                Matrix::copyDataAcrossDevice(nets[0]->getAllWeights().d(), net->getAllWeights().d());
            }
        }
        //主网络负责保存，判断结束条件
        if (net_id == 0)
        {
            bool stop = false;
            if (epoch_count >= total_epochs)
            {
                stop = true;
            }
            if (test_train && epoch_count - effective_epoch_count > total_epochs / 3)
            {
                LOG("Cannot find effective gradient!\n");
                stop = true;
            }
            if (timer_trained.getElapsedTime() > time_limited)
            {
                LOG("Time limited {} s reached, stop training\n", time_limited);
                stop = true;
            }
            if (!save_format.empty()
                && epoch_count % save_epoch == 0)
            {
                std::string save_name = save_format;
                strfunc::replaceAllSubStringRef(save_name, "{epoch}", std::to_string(epoch_count));
                strfunc::replaceAllSubStringRef(save_name, "{date}", Timer::getNowAsString("%F"));
                strfunc::replaceAllSubStringRef(save_name, "{time}", Timer::getNowAsString("%T"));
                //convert::replaceAllSubStringRef(save_name, "{accurary}", std::to_string(test_accuracy_));
                save_name = filefunc::toLegalFilename(save_name);
                net->saveWeight(save_name, makeSaveSign());
            }
            if (stop)
            {
                break;
            }
        }
    }
    train_info.stop = 1;
    if (net_id == 0)
    {
        ConsoleControl::setColor(CONSOLE_COLOR_LIGHT_RED);
        if (test_test_origin)
        {
            LOG("Maximum accuracy on original test set is {:5.2f}% at epoch {}\n",
                max_test_origin_accuracy * 100, max_test_origin_accuracy_epoch);
        }
        if (test_test)
        {
            LOG("Maximum accuracy on transformed test set is {:5.2f}% at epoch {}\n",
                max_test_accuracy * 100, max_test_accuracy_epoch);
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
        net->test("Test on original train set", &data_preparer_->X0, &data_preparer_->Y0, nullptr, force_output, test_type, 0, &result);
    }
    if (option_.getInt("train", "test_train"))
    {
        net->test("Test on transformed train set", &data_preparer_->X, &data_preparer_->Y, nullptr, force_output, test_type, 0, &result);
    }
    if (option_.getInt("train", "test_test_origin"))
    {
        net->test("Test on original test set", &data_preparer2_->X0, &data_preparer2_->Y0, nullptr, force_output, test_type, 0, &result);
    }
    if (option_.getInt("train", "test_test"))
    {
        net->test("Test on transformed test set", &data_preparer2_->X, &data_preparer2_->Y, nullptr, force_output, test_type, 0, &result);
    }
    net->outputNorm();
}

//附加测试集，一般无用
void Brain::extraTest(Net* net, const std::string& section, int force_output /*= 0*/, int test_type /*= 0*/)
{
    if (!option_.hasSection(section))
    {
        return;
    }
    auto dp_test = DataPreparerFactory::makeUniquePtr(&option_, section, net->getX().getDim(), net->getY().getDim());
    net->test("Extra test", &dp_test->X, &dp_test->Y, nullptr, force_output, test_type);
}

int Brain::testExternalData(void* x, void* y, void* a, int n, int attack_times, realc* error)
{
    std::vector<int> n_begin(1, 0), n_count(1, n);
    auto run = [this, x, y, a, attack_times, error, &n_begin, &n_count](int i)
    {
        Matrix X(UnitType::CPU), Y(UnitType::CPU), A(UnitType::CPU);

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

        //auto& m = CudaControl::select(net->getDeviceID())->getMutex();
        //std::lock_guard<std::mutex> lk(m);
        return net->test("", x1, y1, a1, 0, 0, attack_times, error);
    };
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
        std::vector<std::thread> ths(nets_.size());
        for (int i = 1; i < nets_.size(); i++)
        {
            ths[i] = std::thread{ [i, &run]()
                {
                    run(i);
                } };
        }
        run(0);
        //#pragma omp parallel for
        for (int i = 1; i < nets_.size(); i++)
        {
            ths[i].join();
        }
    }
    return 0;
}

}    // namespace cccc