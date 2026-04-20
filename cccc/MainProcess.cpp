#include "MainProcess.h"
#include "ConsoleControl.h"
#include "DynamicLibrary.h"
#include "NetCifa.h"
#include "NetLayer.h"
#include "filefunc.h"
#include "gpu_lib.h"
#include "strfunc.h"
#include <algorithm>
#include <cmath>
#include <thread>

namespace cccc
{

MainProcess::MainProcess()
{
}

MainProcess::~MainProcess()
{
}

//返回为0是正确创建
int MainProcess::init(const std::string& ini /*= ""*/)
{
    LOG("{}\n", Timer::getNowAsString());
    find_gpu_functions();
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

int MainProcess::loadIni(const std::string& ini)
{
    //LOG("Size of real is {}\n", sizeof(real));

    //初始化选项
    //貌似这个设计比较瞎
    if (ini != "")
    {
        if (filefunc::fileExist(ini))
        {
            option_.loadFile(ini);
            train_filename_ = filefunc::getFileMainNameWithoutPath(ini);
        }
        else
        {
            option_.loadString(ini);
        }
    }
    //加载额外的环境变量
    for (auto& kv : option_.getAllKeyValues("env"))
    {
#ifdef _WIN32
        _putenv(std::format("{}={}", kv.key, kv.value).c_str());
#else
        putenv((char*)fmt1::format("{}={}", kv.key, kv.value).c_str());
#endif
    }
    //option_.print();
    batch_ = (std::max)(1, option_.getInt("train", "batch", 100));
    //work_mode_ = option_.getEnum("train", "work_mode", WORK_MODE_NORMAL);
    return 0;
}

int MainProcess::testGPUDevice()
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

    MP_count_ = option_.getInt("train", "mp", 1);
    if (MP_count_ <= 0)
    {
        MP_count_ = 1;
    }
    MP_count_ = 1;    // lite版仅支持单卡
    return 0;
}

int MainProcess::initNets()
{
    Matrix::setCurrentDataType(option_.getEnum<DataType>("train", "data_type", DataType::FLOAT));
    nets_.clear();
    nets_.resize(MP_count_);
    //这里读取ini中指定的device顺序，其中第一个设备为主网络，该值一般来说应由用户指定
    auto mp_device = option_.getVector<int>("train", "mp_device");
    LOG("MP device: {}\n", mp_device);
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
    auto gpu_device_turn = GpuControl::gpuDevicesTurn();
    if (gpu_device_turn.empty())    //无GPU
    {
        mp_device.resize(1);
        mp_device[0] = 0;
    }
    else
    {
        //设备数不正确或者含负数则自动分配
        bool auto_set = mp_device.size() != MP_count_;
        for (int& d : mp_device)
        {
            if (d < 0)
            {
                auto_set = true;
            }
        }
        mp_device.resize(MP_count_);
        for (int i = 0; i < MP_count_; i++)
        {
            if (auto_set)
            {
                mp_device[i] = gpu_device_turn[i % gpu_device_turn.size()];
            }
            else
            {
                mp_device[i] %= gpu_device_turn.size();    //取模，防止越界
            }
        }
    }
    LOG("Reset MP device: {}\n", mp_device);
    gpus_.resize(mp_device.size());
    for (int i = 0; i < mp_device.size(); i++)
    {
        if (mp_device[i] < 0 || mp_device[i] >= gpu_device_turn.size())
        {
            fatalError(std::format("Error: Device {} is not available!! Check \"mp_device\"\n", mp_device[i]));
            return 1;
        }
        gpus_[i].init(mp_device[i]);
        //检查显存是否足够，注意此处实际上是初始化后的，故可能已经有一部分显存已被实例占用
        size_t free_mem, total_mem;
        gpus_[i].getFreeMemory(free_mem, total_mem);
        double need_free = option_.getReal("train", "need_free_mem", 0.6);
        if (1.0 * free_mem / total_mem < need_free)
        {
            fatalError(std::format("Error: Not enough VIDEO memory on device GPU {}!!\nNeed {:.4}%, but only {:.4}% free\n",
                mp_device[i], 100.0 * need_free, 100.0 * free_mem / total_mem));
            return 1;
        }
    }
    auto cifa = option_.getInt("net", "cifa", 1);
    using MYFUNC = Net* (*)();
    auto creator = (MYFUNC)DynamicLibrary::getFunction(option_.getString("net", "library"), option_.getString("net", "function"));
    for (int i = 0; i < MP_count_; i++)
    {
        LOG("Trying to create net {}...\n", i);
        gpus_[i].setAsCurrent();
        std::unique_ptr<Net> net;
        if (creator)
        {
            net = std::unique_ptr<Net>(creator());
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
        net->setId(i);
        int dev_id = net->getGpu()->getDeviceID();
        net->setOption(&option_);
        if (net->init() != 0)
        {
            return 1;
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

    //0号网络加载权值
    if (option_.getInt("train", "load_net") != 0 && !nets_.empty())
    {
        if (nets_[0]->loadWeight(option_.getString("train", "load_file")) < 0)
        {
            fatalError("Error: Load net weight failed!!\n");
            return 1;
        }
    }
    for (int i = 0; i < nets_.size(); i++)
    {
        auto& net = nets_[i];
        //只使用0号网络的权值，随机初始化也如此
        if (i != 0)
        {
            Matrix::copyDataAcrossDevice(nets_[0]->getAllWeights(), net->getAllWeights());
        }
    }
    //主线程使用0号网络
    nets_[0]->setDeviceSelf();
    return 0;
}

int MainProcess::initData()
{
    initDataPreparer();
    initTrainData();
    initTestData();
    return 0;
}

void MainProcess::initDataPreparer()
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
void MainProcess::initTrainData()
{
    if (data_preparer_)
    {
        data_preparer_->prepareData(0, "train");
    }
}

//生成测试集
void MainProcess::initTestData()
{
    if (data_preparer2_)
    {
        data_preparer2_->prepareData(0, "test");
    }
}

std::string MainProcess::makeSaveName(const std::string& save_format, int epoch_count, bool is_filename)
{
    std::string save_name = save_format;
    strfunc::replaceAllSubStringRef(save_name, "{epoch}", std::to_string(epoch_count));
    strfunc::replaceAllSubStringRef(save_name, "{date}", Timer::getNowAsString("%F"));
    strfunc::replaceAllSubStringRef(save_name, "{time}", Timer::getNowAsString("%T"));
    strfunc::replaceAllSubStringRef(save_name, "{filename}", train_filename_);

    while (true)
    {
        auto pos_l = save_name.find("{");
        if (pos_l != std::string::npos)
        {
            auto pos_r = save_name.find("}", pos_l);
            if (pos_r != std::string::npos)
            {
                auto name = save_name.substr(pos_l + 1, pos_r - pos_l - 1);
                auto pos = name.find("::");
                if (pos != std::string::npos)
                {
                    auto section = name.substr(0, pos);
                    auto key = name.substr(pos + 2);
                    auto value = option_.getString(section, key, "");
                    if (is_filename)
                    {
                        value = filefunc::toLegalFilename(value, false);
                    }
                    strfunc::replaceAllSubStringRef(save_name, "{" + name + "}", value);
                }
                else
                {
                    std::string value;
                    for (auto& section : option_.getAllSections())
                    {
                        value = option_.getString(section, name, "");
                        if (value != "")
                        {
                            break;
                        }
                    }
                    if (is_filename)
                    {
                        value = filefunc::toLegalFilename(value, false);
                    }
                    strfunc::replaceAllSubStringRef(save_name, "{" + name + "}", value);
                }
            }
            else
            {
                strfunc::replaceOneSubStringRef(save_name, "{", "");
            }
        }
        else
        {
            break;
        }
    }

    //convert::replaceAllSubStringRef(save_name, "{accurary}", std::to_string(test_accuracy_));
    if (is_filename)
    {
        save_name = filefunc::toLegalFilename(save_name);
    }
    return save_name;
}

std::string MainProcess::makeSaveSign()
{
    std::string sign = option_.getString("train", "save_sign");
    sign = makeSaveName(sign, epoch_count_, false);
    sign = std::format("{}, {}, {}, {}",
        sign,
        Timer::getNowAsString(),
        Timer::autoFormatTime(timer_total_.getElapsedTime()),
        data_preparer_->X.getNumber() * epoch_count_);
    return sign;
}

using std::chrono::operator""ns;
//暂时阻塞等待条件。注意，这里只能用宏，用函数写起来可以用lambda，稍显繁琐
#define WAIT_UNTIL(condition) \
    do { \
        while (!(condition)) { std::this_thread::yield(); } \
    } while (0)

//#define WAIT_UNTIL_OVERTIME(condition, overtime) \
//    do { \
//        auto t0 = std::chrono::system_clock::now(); \
//        while (!(condition) && std::chrono::system_clock::now() - t0 < std::chrono::nanoseconds(int64_t(overtime))) \
//        { \
//            std::this_thread::sleep_for(std::chrono::nanoseconds(100)); \
//        } \
//    } while (0)

//训练一批数据，输出步数和误差
//首个参数为指定几个结构完全相同的网络并行训练
void MainProcess::train(std::vector<std::unique_ptr<Net>>& nets, DataPreparer* data_preparer, int total_epochs)
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

    float e = 0, e0 = 0;
    int test_test = option_.getInt("train", "test_test", 0);
    int test_test_origin = option_.getInt("train", "test_test_origin", 0);
    int test_epoch = option_.getInt("train", "test_epoch", 1);

    TrainInfo ti;
    ti.data_prepared = -1;
    ti.stop = 0;

    //创建训练线程
    std::vector<std::thread> net_threads(nets.size());
    for (int i = 0; i < net_threads.size(); i++)
    {
        net_threads[i] = std::thread{ [this, &nets, i, &ti, total_epochs]()
            {
                trainOneNet(nets, i, ti, total_epochs);
            } };
    }

    for (int epoch_count = 0;; epoch_count++)
    {
        data_preparer->prepareData(epoch_count, "train");
        if (data_preparer2_ && (test_test || test_test_origin) && epoch_count % test_epoch == test_epoch - 1 && data_preparer2_->isFill())
        {
            data_preparer2_->prepareData(epoch_count, "test");
        }
        ti.data_prepared = epoch_count;
        WAIT_UNTIL(ti.data_distributed == MP_count_ || ti.stop);
        ti.data_distributed = 0;
        //回调
        if (running_callback_) { running_callback_(this); }
        iter_count_ += iter_per_epoch;
        epoch_count_++;
        if (ti.stop)
        {
            break;
        }
    }

    ti.stop = 1;

    for (int i = 0; i < net_threads.size(); i++)
    {
        net_threads[i].join();
    }

    LOG("\n");
    data_preparer->summary();
}

//训练网络数组nets中的一个
//在并行时需注意batch的设置，可以认为batch（非并行） = batch（并行）* 线程数
//因分发需要，应传入所有网络
void MainProcess::trainOneNet(std::vector<std::unique_ptr<Net>>& nets, int net_id, TrainInfo& ti, int total_epochs)
{
    auto& net = nets[net_id];
    net->setDeviceSelf();
    net->getGpu()->setActivePhase(ACTIVE_PHASE_TRAIN);

    auto x_dim = data_preparer_->X.getDim();
    auto y_dim = data_preparer_->Y.getDim();
    x_dim.back() /= MP_count_;
    y_dim.back() /= MP_count_;
    Matrix X_train_gpu(x_dim), Y_train_gpu(y_dim), A_train_gpu(y_dim);
    std::unordered_map<std::string, Matrix> extra_train_gpu;
    for (auto& [name, m] : data_preparer_->getExtraData())
    {
        auto dim = m.getDim();
        dim.back() /= MP_count_;
        extra_train_gpu[name] = Matrix(dim);
        dim.back() = net->getBatch();
        net->addExtraMatrix(name, dim);
    }

    int test_epoch = option_.getInt("train", "test_epoch", 1);
    int pre_test_train = option_.getInt("train", "pre_test_train", 0);

    int save_epoch = option_.getInt("train", "save_epoch", 10);
    int save_solver_state = option_.getInt("train", "save_solver_state", 0);
    int out_iter = option_.getInt("train", "out_iter", 100);

    int stop_no_effective_grad = option_.getInt("train", "stop_no_effective_grad", 1);

    std::string save_format = option_.getString("train", "save_format", "save/save-{epoch}.txt");
    int total_batch = data_preparer_->X.getNumber();

    float max_test_accuracy = 0;
    int max_test_accuracy_epoch = 0;

    int iter_count = 0;
    int epoch_count = 0;

    //时间计算相关
    double time_limited = option_.getReal("train", "time_limited", 86400 * 100);
    Timer timer_trained, timer_per_epoch;
    double time_test_total = 0;
    int test_count = 0;

    Matrix workspace;    //同步使用

    Matrix weight_backup(net->getAllWeights().getDim());    //备份权重
    Matrix::copyData(net->getAllWeights(), weight_backup);
    Matrix::copyData(net->getAllWeights().d(), weight_backup.d());

    float l1p = 0, l2p = 0;
    if (net_id == 0)
    {
        int n;
        net->calNorm(n, l1p, l2p);
    }
    int restore_count = 0;
    int failed_count = 0;
    bool is_effective = false;
    double increase_limited = 0;
    std::vector<TestInfo> group_result_max, group_result_prev;    //只保存分组测试结果
    std::vector<std::vector<TestInfo>> all_group_result;          //保存所有的测试结果
    int effective_epoch_count = 0;                                //有效epoch计数，注意仅主网络的这个值在更新

    while (ti.stop == 0)
    {
        //等待数据准备完成
        Timer timer_copy;
        WAIT_UNTIL(ti.data_prepared == epoch_count || ti.stop);
        epoch_count++;
        net->setEpoch(epoch_count);
        if (data_preparer_->is_new_data_)
        {
            Matrix::copyRows(data_preparer_->X, net_id * total_batch / MP_count_, X_train_gpu, 0, X_train_gpu.getNumber());
            Matrix::copyRows(data_preparer_->Y, net_id * total_batch / MP_count_, Y_train_gpu, 0, Y_train_gpu.getNumber());
            for (auto& [name, m] : data_preparer_->getExtraData())
            {
                Matrix::copyRows(m, net_id * total_batch / MP_count_, extra_train_gpu[name], 0, extra_train_gpu[name].getNumber());
            }
        }
        //如果将数据复制放到数据准备线程，则计算线程也会变慢
        LOG("Data for epoch {} copied in {} s\n", epoch_count, timer_copy.getElapsedTime());
        //发出拷贝数据结束信号
        ++ti.data_distributed;
        if (net_id == 0)
        {
            //调整学习率
            //按时间算的奇怪参数
            int progress_by_time = timer_trained.getElapsedTime() / time_limited * total_epochs;
            if (progress_by_time < epoch_count)
            {
                net->solverAdjustLearnRate(epoch_count, total_epochs);
            }
            else
            {
                net->solverAdjustLearnRate(progress_by_time, total_epochs);
            }
            LOG("Learn rate for epoch {} is {}\n", epoch_count, net->getSolver().getLearnRate());
            //训练前在训练集上的测试，若训练集实时生成可以使用
            if (epoch_count % test_epoch == 0 && pre_test_train)
            {
                LOG("Test net {}: before training\n", net_id);
                net->test(&X_train_gpu, &Y_train_gpu, &A_train_gpu);
            }
        }
        Timer timer_net;
        net->clearTime();
        for (int iter = 0; iter < X_train_gpu.getNumber() / net->getX().getNumber(); iter++)
        {
            iter_count++;
            bool output = (iter + 1) % out_iter == 0;
            net->getX().shareData(X_train_gpu, 0, iter * net->getX().getNumber());
            net->getY().shareData(Y_train_gpu, 0, iter * net->getY().getNumber());
            for (auto& [name, m] : extra_train_gpu)
            {
                net->getMatrixByName(name)->shareData(m, 0, iter * net->getMatrixByName(name)->getNumber());
            }
            //WAIT_UNTIL(ti.dweight_uncollected == 0 || ti.stop);
            float e = 0;
            net->active(nullptr, nullptr, nullptr, true, output ? &e : nullptr);
            //网络反向结束，试图通知主网络
            ++ti.trained;

            if (net_id == 0)
            {
                //主网络，完成信息输出，参数的收集和重新分发
                if (output)
                {
                    LOG("epoch = {}, iter = {}, error = {:.8}\n", epoch_count, iter_count, e);
                }
                //主网络等待所有网络反向完成
                WAIT_UNTIL(ti.trained == MP_count_ || ti.stop);
                ti.trained = 0;
                //同步
                if (MP_count_ > 1)
                {
                    auto calSum = [&workspace, &net, &nets, this](auto func)
                    {
                        workspace.resize(func(net).getDim());
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
                    WAIT_UNTIL(ti.data_prepared == epoch_count || ti.stop);    //需要使用到最新的数据，故等待数据准备完成
                    //LOG("Epoch {} finished\n", epoch_count);
                    if (epoch_count % test_epoch == 0)
                    {
                        //测试开始
                        Timer timer_test;
                        testData(net.get(), epoch_count, total_epochs, &max_test_accuracy_epoch, &max_test_accuracy, &all_group_result, &X_train_gpu, &Y_train_gpu);
                        if (net->solverAdjustLearnRate2(epoch_count, total_epochs, all_group_result))
                        {
                            ti.stop = 1;
                        }
                        int n;
                        float l1, l2;
                        net->calNorm(n, l1, l2);
                        LOG("N = {}, L1 = {}, L2 = {}\n", n, l1, l2);
                        //限制权重的增长
                        if (increase_limited == 0 && l2p != 0)
                        {
                            increase_limited = l2 / l2p * 1.5;    //参考首次的变化
                            LOG("Increase limited is {}\n", increase_limited);
                        }
                        //当训练集测试准确率达到一定水平后，认为训练开始生效
                        if (is_effective == false && epoch_count > test_epoch)
                        {
                            is_effective = checkTestEffective(group_result_max, net->getGroupTestInfo());
                        }

                        if (checkTrainHealth(group_result_prev, net->getGroupTestInfo(), l1p, l2p, l1, l2, increase_limited, effective_epoch_count))
                        {
                            Matrix::copyData(net->getAllWeights(), weight_backup);
                            Matrix::copyData(net->getAllWeights().d(), weight_backup.d());
                            l1p = l1;
                            l2p = l2;
                            group_result_prev = net->getGroupTestInfo();
                            failed_count = 0;
                            if (is_effective > 0)
                            {
                                effective_epoch_count += test_epoch;
                            }
                        }
                        else
                        {
                            //测试失败，回退
                            LOG("Numerical fault! Load previous weight!\n");
                            restore_count++;
                            failed_count++;
                            Matrix::copyData(weight_backup, net->getAllWeights());
                            Matrix::copyData(weight_backup.d(), net->getAllWeights().d());
                            if (failed_count >= 5)
                            {
                                //连续错误过多，重置梯度
                                net->getAllWeights().d().fillData(0);
                                failed_count = 0;
                            }
                            net->solverReset();
                            //train_info.need_reset = MP_count_ - 1;
                            //net->checkNorm();
                        }
                        LOG("Effective epoch {}, restore {}\n", effective_epoch_count, restore_count);
                        test_count++;
                        time_test_total += timer_test.getElapsedTime();
                    }
                }
                //收集梯度完成，开始下一轮梯度收集的通知
                auto uncollected = MP_count_ - 1;
                if (uncollected == 0)
                {
                    //只有一个网络的情况，无需等待副网络
                }
                else
                {
                    WAIT_UNTIL(ti.dweight_uncollected == 0 || ti.stop);
                    ti.dweight_uncollected = uncollected;
                }
            }
            else
            {
                //非主网络
                //等待梯度更新结束
                WAIT_UNTIL(ti.dweight_uncollected > 0 || ti.stop);
                --ti.dweight_uncollected;
                //更新到自己的权重
                Matrix::copyDataAcrossDevice(nets[0]->getAllWeights(), net->getAllWeights());
                Matrix::copyDataAcrossDevice(nets[0]->getAllWeights().d(), net->getAllWeights().d());
                //for (int i = 0; i < net->getSolverMatrix().size(); i++)
                //{
                //    Matrix::copyDataAcrossDevice(nets[0]->getSolverMatrix()[i], net->getSolverMatrix()[i]);
                //}
            }
        }
        LOG("Data for epoch {} trained in {} s\n", epoch_count, timer_net.getElapsedTime());
        //主网络负责保存，判断结束条件
        if (net_id == 0)
        {
            //net->outputTime();
            bool stop = false;
            if (epoch_count >= total_epochs)
            {
                stop = true;
            }
            if (stop_no_effective_grad && epoch_count - effective_epoch_count > total_epochs / 3)
            {
                LOG("Cannot find effective gradient!\n");
                stop = true;
            }
            if (timer_trained.getElapsedTime() > time_limited)
            {
                LOG("Time limited {} s reached, stop training\n", time_limited);
                stop = true;
            }
            if (!save_format.empty() && epoch_count % save_epoch == 0)
            {
                net->saveWeight(makeSaveName(save_format, epoch_count), makeSaveSign(), save_solver_state);
            }
            LOG("{} s for this epoch, {} elapsed totally\n",
                timer_per_epoch.getElapsedTime(), Timer::autoFormatTime(timer_total_.getElapsedTime()));
            int total_test = total_epochs / test_epoch;
            double train_time = timer_trained.getElapsedTime() - time_test_total;
            double rest_time = train_time / epoch_count * (total_epochs - epoch_count) + time_test_total / total_test * (total_test - test_count);
            LOG("About {} left\n", Timer::autoFormatTime(rest_time));
            timer_per_epoch.start();
            if (stop)
            {
                break;
            }
        }
        if (epoch_count % test_epoch == 0)
        {
            net->getGpu()->printState();
        }
    }
    ti.stop = 1;
    if (net_id == 0)
    {
        ConsoleControl::setColor(CONSOLE_COLOR_RED);
        LOG("Maximum accuracy on transformed test set is {:5.2f}% at epoch {}\n", max_test_accuracy * 100, max_test_accuracy_epoch);
        if (!save_format.empty())
        {
            net->saveWeight(makeSaveName(save_format, epoch_count), makeSaveSign(), save_solver_state);
        }
        ConsoleControl::resetColor();
    }
}

bool MainProcess::checkTestEffective(std::vector<TestInfo>& group_result_max, const std::vector<TestInfo>& group_result)
{
    //有效epoch计数，所有非NAN的准确率都曾经大于0.1，则认为训练开始有效果
    if (group_result.empty())
    {
        //特别情况
        return true;
    }
    if (group_result_max.empty() && !group_result.empty())
    {
        group_result_max = group_result;
    }
    int not_nan = 0, more_than_10 = 0;
    for (int i = 0; i < group_result.size(); i++)
    {
        if (!std::isnan(group_result_max[i].accuracy) && !std::isnan(group_result[i].accuracy))
        {
            not_nan++;
            group_result_max[i].accuracy = std::max(group_result_max[i].accuracy, group_result[i].accuracy);
            if (group_result_max[i].accuracy > 0.1)
            {
                more_than_10++;
            }
        }
    }
    if (not_nan > 0 && more_than_10 == not_nan)
    {
        return true;
    }
    return false;
}

bool MainProcess::checkTrainHealth(const std::vector<TestInfo>& group_result_prev, const std::vector<TestInfo>& group_result, float l1p, float l2p, float l1, float l2, double increase_limited, int effective_epoch_count)
{
    //返回真表示没有问题
    auto checkResultV = [](const std::vector<TestInfo>& group_result) -> bool
    {
        //若准确率只有1或0，则表示该结果有问题
        //但若全是1，则没有问题
        if (group_result.size() == 0)
        {
            return true;
        }
        int count0 = 0, count1 = 0;
        for (int i = 0; i < group_result.size(); i++)
        {
            if (group_result[i].right == group_result[i].total)    //若total为0也视为正确
            {
                count1++;
            }
            else if (group_result[i].right == 0)
            {
                count0++;
            }
        }
        if (count0 > 0 && count0 + count1 == group_result.size())
        {
            return false;
        }
        return true;
    };
    //返回真表示没有问题
    auto checkTestError = [checkResultV, effective_epoch_count, &group_result]() -> bool
    {
        //训练未成功，则暂时为真
        //结果正常即为真
        return effective_epoch_count == 0 || checkResultV(group_result);
    };
    //返回真表示没有问题
    auto checkAccurBack = [&group_result, &group_result_prev]() -> bool
    {
        //若准确率在达到一定水平后忽然降低过多则认为失败
        int count = 0;
        for (int i = 0; i < std::min(group_result.size(), group_result_prev.size()); i++)
        {
            if (!std::isnan(group_result[i].accuracy) && !std::isnan(group_result_prev[i].accuracy)
                && group_result_prev[i].accuracy > 0.1 && group_result[i].accuracy / group_result_prev[i].accuracy < 0.1)
            {
                count++;
            }
        }
        return count == 0;
    };
    //返回真表示没有问题
    auto checkNetWeights = [l1p, l2p, l1, l2, increase_limited]() -> bool
    {
        if (l1p == 0 || l2p == 0)
        {
            return true;
        }
        //权重变化不能太剧烈
        return !std::isnan(l1) && !std::isnan(l2) && l1 / l1p < increase_limited && l2 / l2p < increase_limited;    // && l2 < l1;   暂时去掉这个条件
    };
    //返回真表示没有问题
    return checkTestError() && checkNetWeights() && checkAccurBack();
}

//运行，注意容错保护较弱
//注意通常情况下是使用第一个网络测试数据
void MainProcess::run()
{
    auto& net = nets_[0];

    //初测
    testData(net.get());

    train(nets_, data_preparer_.get(), option_.getInt("train", "train_epochs", 10));

    std::string save_filename = option_.getString("train", "save_file");
    if (save_filename != "")
    {
        net->saveWeight(save_filename, makeSaveSign(), option_.getInt("train", "save_solver_state"));
    }
    //net->getSolver().outputState();
    //终测
    testData(net.get());

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

//输出训练集和测试集的测试结果
void MainProcess::testData(Net* net, int epoch, int total_epochs, int* max_accuracy_epoch, float* max_accuracy, std::vector<std::vector<TestInfo>>* collect_test_info, Matrix* X, Matrix* Y)
{
    if (net == nullptr)
    {
        net = nets_[0].get();
    }
    //ata_preparer_->X0.message("Train data X0");
    //data_preparer_->X.message("Train data X");
    if (option_.getInt("train", "test_train_origin"))
    {
        LOG("Test net {}: original train set\n", net->getId());
        Matrix temp(DataType::CURRENT, UnitType::CPU);
        net->test(&data_preparer_->X0, &data_preparer_->Y0, &temp);
        if (epoch >= total_epochs * 0.833)
        {
            data_preparer_->checkData(temp);
        }
    }
    if (option_.getInt("train", "test_train"))
    {
        LOG("Test net {}: transformed train set\n", net->getId());
        Matrix temp(DataType::CURRENT, UnitType::CPU);
        //注意其实变换后的训练集应该在显存中，且训练中必定存在，因此可以通过参数传入
        if (X && Y)
        {
            net->test(X, Y, &temp);
        }
        else
        {
            net->test(&data_preparer_->X, &data_preparer_->Y, &temp);
        }
        //if (epoch >= total_epochs * 0.833)
        {
            data_preparer_->checkData(temp);
        }
        if (collect_test_info)
        {
            collect_test_info->push_back(net->getGroupTestInfo());
        }
    }
    if (data_preparer2_)
    {
        if (option_.getInt("train", "test_test_origin"))
        {
            LOG("Test net {}: original test set\n", net->getId());
            Matrix temp(DataType::CURRENT, UnitType::CPU);
            net->test(&data_preparer2_->X0, &data_preparer2_->Y0, &temp);
            data_preparer2_->checkData(temp);
            if (max_accuracy_epoch && max_accuracy)
            {
                if (net->getTestInfo().accuracy >= *max_accuracy)
                {
                    *max_accuracy = net->getTestInfo().accuracy;
                    *max_accuracy_epoch = epoch;
                }
            }
        }
        if (option_.getInt("train", "test_test"))
        {
            LOG("Test net {}: transformed test set\n", net->getId());
            Matrix temp(DataType::CURRENT, UnitType::CPU);
            net->test(&data_preparer2_->X, &data_preparer2_->Y, &temp);
            data_preparer2_->checkData(temp);
            if (max_accuracy_epoch && max_accuracy)
            {
                if (net->getTestInfo().accuracy >= *max_accuracy)
                {
                    *max_accuracy = net->getTestInfo().accuracy;
                    *max_accuracy_epoch = epoch;
                }
            }
        }
    }
}

//附加测试集，一般无用
void MainProcess::extraTest(Net* net, const std::string& section, int force_output /*= 0*/, int test_type /*= 0*/)
{
    if (!option_.hasSection(section))
    {
        return;
    }
    auto dp_test = DataPreparerFactory::makeUniquePtr(&option_, section, net->getX().getDim(), net->getY().getDim());
    LOG("Extra test\n");
    net->test(&dp_test->X, &dp_test->Y, nullptr);
}

int MainProcess::testExternalData(void* x, void* y, void* a, int n, int attack_times, double* error)
{
    std::vector<int> n_begin(1, 0), n_count(1, n);
    auto run = [this, x, y, a, attack_times, error, &n_begin, &n_count](int i)
    {
        auto net = nets_[i].get();
        auto dt = net->getX().getDataType();
        Matrix X(dt, UnitType::CPU), Y(dt, UnitType::CPU), A(dt, UnitType::CPU);
        auto dim0 = net->getX().getDim();
        dim0.back() = n_count[i];
        auto dim1 = net->getY().getDim();
        dim1.back() = n_count[i];
        auto size = X.getDataTypeSize();

        Matrix *x1 = nullptr, *y1 = nullptr, *a1 = nullptr;
        if (x)
        {
            X.shareData((char*)x + net->getX().getRow() * n_begin[i] * size);
            X.resize(dim0);
            //X = X.transDataType(DataType::HALF);
            x1 = &X;
        }
        if (y)
        {
            Y.shareData((char*)y + net->getY().getRow() * n_begin[i] * size);
            Y.resize(dim1);
            y1 = &Y;
        }
        if (a)
        {
            A.shareData((char*)a + net->getA().getRow() * n_begin[i] * size);
            A.resize(dim1);
            a1 = &A;
        }

        //auto& m = CudaControl::select(net->getDeviceID())->getMutex();
        //std::lock_guard<std::mutex> lk(m);
        return net->test_only(X, A);
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
        for (int i = 1; i < nets_.size(); i++)
        {
            ths[i].join();
        }
    }
    return 0;
}

}    // namespace cccc