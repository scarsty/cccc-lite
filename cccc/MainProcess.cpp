#include "MainProcess.h"
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

MainProcess::MainProcess()
{
}

MainProcess::~MainProcess()
{
}

//返回为0是正确创建
int MainProcess::init(const std::string& ini /*= ""*/)
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

    MP_count_ = 1;
    if (MP_count_ <= 0)
    {
        MP_count_ = 1;
    }
    return 0;
}

int MainProcess::initNets()
{
    Matrix::setCurrentDataType(option_.getEnum<DataType>("train", "data_type", DataType::FLOAT));
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
        if (mp_device[i] < 0 || mp_device[i] >= MP_count_)
        {
            fatalError(fmt1::format("Error: Device {} is not available!! Check \"mp_device\"\n", mp_device[i]));
            return 1;
        }
        gpus_[i].init(mp_device[i]);
        //检查显存是否足够，注意此处实际上是初始化后的，故可能已经有一部分显存已被实例占用
        size_t free_mem, total_mem;
        gpus_[i].getFreeMemory(free_mem, total_mem);
        double need_free = option_.getReal("train", "need_free_mem", 0.6);
        if (1.0 * free_mem / total_mem < need_free)
        {
            fatalError(fmt1::format("Error: Not enough memory on device {}!!\nNeed {}%, but only {}% free\n",
                mp_device[i], 100.0 * need_free, 100.0 * free_mem / total_mem));
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
        int dev_id = net->getGpu()->getDeviceID();
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

std::string MainProcess::makeSaveName(const std::string& save_format, int epoch_count)
{
    std::string save_name = save_format;
    strfunc::replaceAllSubStringRef(save_name, "{epoch}", std::to_string(epoch_count));
    strfunc::replaceAllSubStringRef(save_name, "{date}", Timer::getNowAsString("%F"));
    strfunc::replaceAllSubStringRef(save_name, "{time}", Timer::getNowAsString("%T"));
    //convert::replaceAllSubStringRef(save_name, "{accurary}", std::to_string(test_accuracy_));
    save_name = filefunc::toLegalFilename(save_name);
    return save_name;
}

std::string MainProcess::makeSaveSign()
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
void MainProcess::run(int train_epochs /*= -1*/)
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
void MainProcess::trainOneNet(std::vector<std::unique_ptr<Net>>& nets, int net_id, TrainInfo& train_info, int total_epochs)
{
    auto& net = nets[net_id];
    net->setDeviceSelf();
    net->getGpu()->setActivePhase(ACTIVE_PHASE_TRAIN);

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

    int stop_no_effective_grad = option_.getInt("train", "stop_no_effective_grad", 1);

    std::string save_format = option_.dealString(option_.getString("train", "save_format", "save/save-{epoch}.txt"), false);
    int total_batch = data_preparer_->X.getNumber();

    float max_test_origin_accuracy = 0;
    int max_test_origin_accuracy_epoch = 0;
    float max_test_accuracy = 0;
    int max_test_accuracy_epoch = 0;

    int iter_count = 0;
    int epoch_count = 0;

    //时间计算相关
    double time_limited = option_.getReal("train", "time_limited", 86400 * 100);
    Timer timer_test, timer_trained;
    double time_test0 = 0;
    double time_per_epoch = 1;

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
    std::vector<TestInfo> resultv_max, resultvp;
    std::vector<std::vector<TestInfo>> all_resultv;
    int effective_epoch_count = 0;    //有效epoch计数，注意仅主网络的这个值在更新

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
        if (net_id == 0)
        {
            //调整学习率
            //按时间算的奇怪参数
            int progress_by_time = timer_trained.getElapsedTime() / time_limited * total_epochs;
            if (progress_by_time < epoch_count)
            {
                net->getSolver().adjustLearnRate(epoch_count, total_epochs);
            }
            else
            {
                net->getSolver().adjustLearnRate(progress_by_time, total_epochs);
            }
            LOG("Learn rate for epoch {} is {}\n", epoch_count, net->getSolver().getLearnRate());
        }
        //训练前在训练集上的测试，若训练集实时生成可以使用
        if (net_id == 0 && epoch_count % test_epoch == 0 && pre_test_train)
        {
            net->test(&X_train_gpu, &Y_train_gpu, &A_train_gpu, "Test on train set before training", 0, 1);
        }
        float e = 0;
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
            //发出网络反向结束信号
            train_info.trained++;

            if (net_id == 0)
            {
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
                    //LOG("Epoch {} finished\n", epoch_count);
                    if (epoch_count % test_epoch == 0)
                    {
                        std::string content = fmt1::format("Test net {}: ", net_id);
                        double test_result;
                        std::vector<TestInfo> resultv;
                        //int test_error = 0;
                        if (test_train_origin)
                        {
                            net->test(&data_preparer_->X, &data_preparer_->Y, nullptr, content + "original train set", 0, test_type, 0, &test_result);
                        }
                        if (test_train)
                        {
                            std::vector<std::vector<TestInfo>> resultvv;
                            net->test(&X_train_gpu, &Y_train_gpu, nullptr, content + "transformed train set", 0, test_type, 0, &test_result, &resultvv);    //有返回分开的准确率
                            if (!resultvv.empty())
                            {
                                //注意后续的判断只考虑第一组
                                resultv = resultvv[0];
                                all_resultv.push_back(resultv);
                            }
                        }
                        if (data_preparer2_)
                        {
                            if (test_test_origin)
                            {
                                net->test(&data_preparer2_->X0, &data_preparer2_->Y0, nullptr, content + "original test set", 0, test_type, 0, &test_result);
                                if (test_result >= max_test_origin_accuracy)
                                {
                                    max_test_origin_accuracy = test_result;
                                    max_test_origin_accuracy_epoch = epoch_count;
                                }
                            }
                            if (test_test)
                            {
                                net->test(&data_preparer2_->X, &data_preparer2_->Y, nullptr, content + "transformed test set", 0, test_type, 0, &test_result);
                                if (test_result >= max_test_accuracy)
                                {
                                    max_test_accuracy = test_result;
                                    max_test_accuracy_epoch = epoch_count;
                                }
                            }
                        }

                        if (net->getSolver().adjustLearnRate2(epoch_count, total_epochs, all_resultv))
                        {
                            train_info.stop = 1;
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
                            is_effective = checkTestEffective(resultv_max, resultv);
                        }

                        if (checkTrainHealth(resultvp, resultv, l1p, l2p, l1, l2, increase_limited, effective_epoch_count))
                        {
                            Matrix::copyData(net->getAllWeights(), weight_backup);
                            Matrix::copyData(net->getAllWeights().d(), weight_backup.d());
                            l1p = l1;
                            l2p = l2;
                            resultvp = resultv;
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
                            net->getSolver().reset();
                            //train_info.need_reset = MP_count_ - 1;
                            //net->checkNorm();
                        }

                        auto time_test = timer_test.getElapsedTime();
                        time_per_epoch = (time_test - time_test0) / test_epoch;
                        time_test0 = time_test;
                        double time_rest;
                        time_rest = (total_epochs - epoch_count) * time_per_epoch;
                        time_rest = std::min(time_rest, time_limited - timer_trained.getElapsedTime());
                        time_rest = std::max(time_rest, 0.0);
                        net->getSolver().outputState();
                        LOG("Effective epoch {}, restore {}\n", effective_epoch_count, restore_count);
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
                //for (int i = 0; i < net->getSolverMatrix().size(); i++)
                //{
                //    Matrix::copyDataAcrossDevice(nets[0]->getSolverMatrix()[i], net->getSolverMatrix()[i]);
                //}
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
            if (stop_no_effective_grad && test_train && epoch_count - effective_epoch_count > total_epochs / 3)
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
                net->saveWeight(makeSaveName(save_format, epoch_count), makeSaveSign());
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
        if (!save_format.empty())
        {
            net->saveWeight(makeSaveName(save_format, epoch_count), makeSaveSign());
        }
        ConsoleControl::setColor(CONSOLE_COLOR_NONE);
    }
}

bool MainProcess::checkTestEffective(std::vector<TestInfo>& resultv_max, std::vector<TestInfo>& resultv)
{
    //有效epoch计数，所有非NAN的准确率都曾经大于0.1，则认为训练开始有效果
    if (resultv.empty())
    {
        //特别情况
        return true;
    }
    if (resultv_max.empty() && !resultv.empty())
    {
        resultv_max = resultv;
    }
    int not_nan = 0, more_than_10 = 0;
    for (int i = 0; i < resultv.size(); i++)
    {
        if (!std::isnan(resultv_max[i].accuracy) && !std::isnan(resultv[i].accuracy))
        {
            not_nan++;
            resultv_max[i].accuracy = std::max(resultv_max[i].accuracy, resultv[i].accuracy);
            if (resultv_max[i].accuracy > 0.1)
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

bool MainProcess::checkTrainHealth(const std::vector<TestInfo>& resultvp, const std::vector<TestInfo>& resultv, float l1p, float l2p, float l1, float l2, double increase_limited, int effective_epoch_count)
{
    //返回真表示有问题
    auto checkResultV = [](const std::vector<TestInfo>& resultv) -> bool
    {
        //若准确率只有1或0，则表示该结果有问题
        //但若全是1，则没有问题
        if (resultv.size() == 0)
        {
            return false;
        }
        int count0 = 0, count1 = 0;
        for (int i = 0; i < resultv.size(); i++)
        {
            if (resultv[i].right == resultv[i].total)    //若total为0也视为正确
            {
                count1++;
            }
            else if (resultv[i].right == 0)
            {
                count0++;
            }
        }
        if (count0 > 0 && count0 + count1 == resultv.size())
        {
            return true;
        }
        return false;
    };
    //以下返回真表示没有问题
    auto checkTestError = [checkResultV, effective_epoch_count, &resultv]() -> bool
    {
        //在训练已经生效的情况下测试失败
        return !(effective_epoch_count > 0 && checkResultV(resultv));
    };
    auto checkAccurBack = [&resultv, &resultvp]() -> bool
    {
        //若准确率在达到一定水平后忽然降低过多则认为失败
        int count = 0;
        for (int i = 0; i < std::min(resultv.size(), resultvp.size()); i++)
        {
            if (!std::isnan(resultv[i].accuracy) && !std::isnan(resultvp[i].accuracy)
                && resultvp[i].accuracy > 0.1 && resultv[i].accuracy / resultvp[i].accuracy < 0.1)
            {
                count++;
            }
        }
        return count == 0;
    };
    auto checkNetWeights = [l1p, l2p, l1, l2, increase_limited]() -> bool
    {
        if (l1p == 0 || l2p == 0)
        {
            return true;
        }
        //权重变化不能太剧烈
        return !std::isnan(l1) && !std::isnan(l2) && l1 / l1p < increase_limited && l2 / l2p < increase_limited && l2 < l1;
    };
    return checkTestError() && checkNetWeights() && checkAccurBack();
}

//输出训练集和测试集的测试结果
void MainProcess::testData(Net* net, int force_output /*= 0*/, int test_type /*= 0*/)
{
    if (net == nullptr)
    {
        net = nets_[0].get();
    }
    //ata_preparer_->X0.message("Train data X0");
    //data_preparer_->X.message("Train data X");
    if (option_.getInt("train", "test_train_origin"))
    {
        net->test(&data_preparer_->X0, &data_preparer_->Y0, nullptr, "Test on original train set", force_output, test_type);
    }
    if (option_.getInt("train", "test_train"))
    {
        net->test(&data_preparer_->X, &data_preparer_->Y, nullptr, "Test on transformed train set", force_output, test_type);
    }
    if (option_.getInt("train", "test_test_origin"))
    {
        net->test(&data_preparer2_->X0, &data_preparer2_->Y0, nullptr, "Test on original test set", force_output, test_type);
    }
    if (option_.getInt("train", "test_test"))
    {
        net->test(&data_preparer2_->X, &data_preparer2_->Y, nullptr, "Test on transformed test set", force_output, test_type);
    }
    net->outputNorm();
}

//附加测试集，一般无用
void MainProcess::extraTest(Net* net, const std::string& section, int force_output /*= 0*/, int test_type /*= 0*/)
{
    if (!option_.hasSection(section))
    {
        return;
    }
    auto dp_test = DataPreparerFactory::makeUniquePtr(&option_, section, net->getX().getDim(), net->getY().getDim());
    net->test(&dp_test->X, &dp_test->Y, nullptr, "Extra test", force_output, test_type);
}

int MainProcess::testExternalData(void* x, void* y, void* a, int n, int attack_times, double* error)
{
    std::vector<int> n_begin(1, 0), n_count(1, n);
    auto run = [this, x, y, a, attack_times, error, &n_begin, &n_count](int i)
    {
        Matrix X(DataType::FLOAT, UnitType::CPU), Y(DataType::FLOAT, UnitType::CPU), A(DataType::FLOAT, UnitType::CPU);

        auto net = nets_[i].get();
        auto dim0 = net->getX().getDim();
        dim0.back() = n_count[i];
        auto dim1 = net->getY().getDim();
        dim1.back() = n_count[i];

        Matrix *x1 = nullptr, *y1 = nullptr, *a1 = nullptr;
        if (x)
        {
            X.shareData((float*)x + net->getX().getRow() * n_begin[i]);
            X.resize(dim0);
            x1 = &X;
        }
        if (y)
        {
            Y.shareData((float*)y + net->getY().getRow() * n_begin[i]);
            Y.resize(dim1);
            y1 = &Y;
        }
        if (a)
        {
            A.shareData((float*)a + net->getA().getRow() * n_begin[i]);
            A.resize(dim1);
            a1 = &A;
        }

        //auto& m = CudaControl::select(net->getDeviceID())->getMutex();
        //std::lock_guard<std::mutex> lk(m);
        return net->test(x1, y1, a1, "", 0, 0, attack_times, error);
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