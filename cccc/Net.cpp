#include "Net.h"
#include "ConsoleControl.h"
#include "CudnnGraph.h"
#include "INIReaderBin.h"
#include "MatrixEx.h"
#include "Timer.h"
#include "VectorMath.h"
#include "filefunc.h"
#include <algorithm>
#include <cstdlib>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace cccc
{

Net::Net()
{
}

Net::~Net()
{
}

int Net::init()
{
    const bool infer_mode = option_->getInt("train", "train_epochs", 0) <= 0;
    // 推理模式下固定开启 lazy_build（先规划复用再分配显存，降低峰值）
    const bool lazy_build = infer_mode && option_->getInt("train", "disable_activation_memory_reuse", 0) == 0;
    if (lazy_build)
    {
        MatrixData::setLazyMode(true);
    }
    int r = init2();
    if (lazy_build)
    {
        MatrixData::setLazyMode(false);
    }
    if (r)
    {
        return r;
    }
    //MatrixOp::checkConnect(op_queue_, getX(), getA());
    if (op_queue_.size() == 0)
    {
        LOG("Empty compute queue!\n");
        r = -2;
    }
    getX().setNeedBack(false);
    //workspace_.resize(all_weights_);
    if (loss_.empty())    //未设置损失函数就自己添加一个，最简单的情况
    {
        MatrixOp op;
        op.set(MatrixOpType::LOSS, { A_, Y_ }, {}, {}, {}, {});
        loss_.push_back(op);
        //LossWeight_->printAsMatrix();
    }

    MatrixOp::checkConnect(op_queue_, getX(), getA(), loss_);
    initWeights();
    // optimizeActivationMemory 已在同一遍历中完成显存分配，无需单独的 materialize 阶段。
    optimizeActivationMemory();    // 推理：跨层激活显存复用 + 立即分配显存，降低显存峰值

    // 仅训练时需要 solver（推理时 all_weights_ 为空，dW_ 不必分配）
    if (option_->getInt("train", "train_epochs", 0) > 0)
    {
        solver_.init(option_, "train", all_weights_);
    }

    separate_update_weight_ = option_->getInt("train", "separate_update_weight", 0);

    float learn_rate_base = option_->getReal("train", "learn_rate_base", 0.01f);
    if (separate_update_weight_)
    {
        for (auto& op : op_queue_)
        {
            for (auto& m : op.getMatrixIn())
            {
                if (m->isWeight())
                {
                    solvers_[m.get()] = std::make_shared<Solver>();
                    auto& s = *solvers_[m.get()];
                    auto learn_rate_base1 = learn_rate_base;
                    if (op.solver_type_ == SOLVER_ADAM)
                    {
                        learn_rate_base1 = learn_rate_base * 0.1f;
                    }
                    option_->setKey("train", "learn_rate_base", std::to_string(learn_rate_base1));
                    option_->setKey("train", "solver", option_->getStringFromEnum(op.solver_type_));
                    s.init(option_, "train", *m);
                }
            }
        }
    }

#if defined(_DEBUG)
    //LOG("{}\n", ir());
    //MatrixOp::ir(loss_);
#endif
    if (getBatch() <= 0)
    {
        resetBatchSize(option_->getInt("train", "batch", 16));
    }

    //计算总占用空间
    //std::map<void*, int64_t> map1;
    //int64_t max_size = 0, sum = 0;
    //for (auto& op : op_queue_)
    //{
    //    for (auto& m : op.getMatrixIn())
    //    {
    //        map1[m->getDataPtr()] = std::max(m->getDataSizeInByte(), map1[m->getDataPtr()]);
    //        max_size = std::max(max_size, m->getDataSizeInByte());
    //    }
    //}
    //for (auto s : map1)
    //{
    //    sum += s.second;
    //}
    //LOG("Total size {:e}, max size {:e}\n", sum * 1.0, max_size * 1.0);

    // 图模式：整网前向一张图，由 cuDNN 自动融合。
    use_graph_ = (option_->getInt("train", "use_cudnn_graph", 0) != 0);
    if (use_graph_)
    {
        buildForwardGraph();
    }

    return r;
}

MatrixSP& Net::getMatrixByName(const std::string& name)
{
    if (!extra_matrixsp_.contains(name))
    {
        extra_matrixsp_[name] = makeMatrixSP();
    }
    return extra_matrixsp_[name];
}

void Net::addExtraMatrix(const std::string& name, const std::vector<int>& dim)
{
    if (extra_matrixsp_.contains(name))
    {
        extra_matrixsp_[name]->resize(dim);
    }
    else
    {
        extra_matrixsp_[name] = makeMatrixSP(dim);
    }
}

//learn为真时，会反向更新网络
//active只处理一个gpu中的minibatch
//A是外部提供的矩阵，用于保存结果
void Net::active(Matrix* X, Matrix* Y, Matrix* A, bool back, float* error)
{
    //setDeviceSelf();
    if (X)
    {
        getX().shareData(*X);
    }
    if (Y)
    {
        getY().shareData(*Y);
    }
    //Timer t0;
    if (use_graph_ && fwd_graph_)
    {
        runForwardWithGraph();
    }
    else
    {
        MatrixOp::forward(op_queue_);
    }
    //LOG("Forward time: {} s\n", t0.getElapsedTime());
    //getX().message("getX");
    //getY().message("getY");
    //getA().message("getA");
    if (back)
    {
        //Timer t1;
        MatrixOp::backward(op_queue_, loss_, true);
        //LOG("Backward time: {} s\n", t1.getElapsedTime());
    }
    if (A)
    {
        Matrix::copyDataPtr(getA(), getA().getDataPtr(), *A, A->getDataPtr(), getA().getDataSizeInByte());
    }
    if (error && getY().getNumber() > 0)
    {
        //此处计算的是loss梯度的平方和除以batch数，用于相对比较训练误差趋势
        //注意这不是标准MSE（未除以输出维度），仅用于训练过程的误差监控
        *error = 0;
        for (auto& l : loss_)
        {
            *error += l.getMatrixIn()[0]->d().dotSelf();
        }
        *error /= getY().getNumber();
    }
}

void Net::updateWeight()
{
    if (separate_update_weight_)
    {
        for (auto& m : weights_)
        {
            solvers_[m]->updateWeights(*m, getX().getNumber());
        }
    }
    else
    {
        solver_.updateWeights(all_weights_, getX().getNumber());
    }
}

//保存权重，需配合ini中的网络结构
//返回值：0正常，其他值不正常
int Net::saveWeight(const std::string& filename, const std::string& sign, int solver_state)
{
    setDeviceSelf();
    if (filename.empty())
    {
        return -1;
    }
    LOG("Save net to {}... ", filename);
    filefunc::makePath(filefunc::getParentPath(filename));

    INIReaderBin file_bin;
    int index = 0;
    //权重分开保存
    for (auto& m : weights_)
    {
        LOG("Saving weight {} size={}\n", index, m->getDataSizeInByte());
        std::string buffer(m->getDataSizeInByte(), '\0');
        m->save(buffer.data(), m->getDataSize());
        file_bin.set_value("weight_bin" + std::to_string(index++), buffer);
    }
    if (!weights_.empty())
    {
        auto& m = weights_[0];
        file_bin.set_value("data_type", option_->getStringFromEnum(m->getDataType()));
    }
    if (!sign.empty())
    {
        file_bin.set_value("save_sign", sign);
    }
    LOG("Save sign: {}\n", sign.c_str());
    if (file_bin.save(filename) > 0)
    {
        LOG("done\n");
        return 0;
    }
    else
    {
        LOG("failed!\n");
        return -1;
    }
}

//载入权重，需配合ini中的网络结构
//load_mode: 0为从文件读，1为从字串读
//返回值：
//0    读入长度恰好为网络尺寸
//1    读入字节数不等，此时通常不会是正常结果，但可能可以继续
//-2   不能读入文件
int Net::loadWeight(const std::string& str, int load_mode, int solver_state)
{
    setDeviceSelf();
    if (str == "")
    {
        ConsoleControl::setColor(CONSOLE_COLOR_RED);
        LOG("Warning: no data!\n");
        ConsoleControl::setColor(CONSOLE_COLOR_NONE);
        return -2;
    }
    if (load_mode == 0)
    {
        LOG("Loading net from {}... ", str);
    }
    else
    {
        LOG("Loading net from memory... ");
    }
    std::string buffer;
    INIReaderBin file_bin;
    if (filefunc::fileExist(str))
    {
        buffer = filefunc::readFileToString(str);
        if (buffer.empty())
        {
            LOG("failed!\n");
            return -2;
        }
        file_bin.parse(buffer);
    }
    else if (load_mode)
    {
        buffer = str;
        if (buffer.empty())
        {
            LOG("failed!\n");
            return -2;
        }
        file_bin.parse(buffer);
    }
    else
    {
        LOG("failed!\n");
        return -2;
    }

    int ret = 0;
    // 用 c_str() 重构字符串，截断内嵌 '\0'（某些转换工具写入 null 填充如 "half\0\0\0\0"）
    auto data_type = option_->getEnumFromString<DataType>(std::string(file_bin.get_value("data_type").c_str()));
    LOG("Data type of save file is {}\n", option_->getStringFromEnum(data_type));
    LOG("Data type of net is {}\n", option_->getStringFromEnum(weights_[0]->getDataType()));
    if (file_bin.has_value("weight_bin0") || option_->getInt("train", "named_weights", 0) != 0
        || !file_bin.get_value("named_weights").empty())
    {
        //分开保存的情况
        //named_weights=1 时：按权重名称 "weight_<name>" 匹配加载，顺序无关
        //named_weights=0 或未设置时：按顺序用 weight_bin<index> 加载（默认行为）
        bool named_load = option_->getInt("train", "named_weights", 0) != 0;
        //若 ini 未设置，也可由 bin 文件自身声明
        if (!named_load)
        {
            auto nw = file_bin.get_value("named_weights");
            named_load = !nw.empty() && nw[0] != '0' && nw[0] != '\0';
        }

        auto load_one = [&](Matrix* m, const std::string& key) -> int
        {
            auto weight_str = file_bin.get_value(key);
            if (weight_str.empty()) { return -1; }
            if (data_type == m->getDataType())
            {
                m->load((char*)weight_str.data(), weight_str.size());
            }
            else
            {
                Matrix m1(m->getDim(), m->getDataType(), UnitType::CPU);
                size_t src_elem_size = MatrixData::getDataTypeSize(data_type);
                for (int64_t i = 0; i < m->getDataSize(); i++)
                {
                    m1.setData(i, (char*)weight_str.data() + i * src_elem_size, data_type);
                }
                Matrix::copyData(m1, *m);
            }
            if (weight_str.size() / MatrixData::getDataTypeSize(data_type) != (size_t)m->getDataSize())
            {
                ConsoleControl::setColor(CONSOLE_COLOR_RED);
                LOG("Warning: weight '{}' needs {} floats, but {} supplied!\n", key, m->getDataSize(), weight_str.size() / MatrixData::getDataTypeSize(data_type));
                ConsoleControl::resetColor();
                return 1;
            }
            return 0;
        };

        int index = 0;
        for (auto& m : weights_)
        {
            if (m->needLoad())
            {
                int r = -1;
                if (named_load && !m->getWeightName().empty())
                {
                    r = load_one(m, "weight_" + m->getWeightName());
                    if (r < 0)
                    {
                        ConsoleControl::setColor(CONSOLE_COLOR_RED);
                        LOG("Warning: named weight '{}' not found, falling back to weight_bin{}\n", m->getWeightName(), index);
                        ConsoleControl::resetColor();
                    }
                }
                if (r < 0)
                {
                    r = load_one(m, "weight_bin" + std::to_string(index));
                }
                if (r != 0) { ret = 1; }
            }
            index++;
        }
    }
    else
    {
        //旧格式，整体保存
        auto weight_str = file_bin.get_value("weight_binary");
        if (weight_str.empty())
        {
            weight_str = buffer;    //直接当成二进制流也可
        }
        int64_t sum = 0;    //已使用的文件中的字节数
        auto p = weight_str.data();
        size_t size1_in_save = MatrixData::getDataTypeSize(data_type);
        for (auto& m : weights_)
        {
            if (m->needLoad())
            {
                if (sum + m->getDataSize() * size1_in_save > weight_str.size())
                {
                    sum += m->getDataSize() * size1_in_save;
                    p += m->getDataSize() * size1_in_save;
                    continue;
                }
                if (data_type == m->getDataType())
                {
                    auto loaded = m->load((char*)weight_str.data() + sum, weight_str.size() - sum);
                    sum += loaded;
                    p += loaded;
                }
                else
                {
                    Matrix m1(m->getDim(), m->getDataType(), UnitType::CPU);
                    for (int64_t i = 0; i < m->getDataSize(); i++)
                    {
                        m1.setData(i, p, data_type);
                        p += size1_in_save;
                        sum += size1_in_save;
                    }
                    Matrix::copyData(m1, *m);
                }
            }
            else
            {
                //跳过不需要加载的数据，例如卷积层不变，全连接层重新训练等
                sum += m->getDataSize() * size1_in_save;
                p += m->getDataSize() * size1_in_save;
                //LOG("Skip {} bytes!\n", m->getDataSize());
            }
        }
        LOG("done\n");
        ConsoleControl::setColor(CONSOLE_COLOR_RED);
        if (weight_str.size() != sum)
        {
            LOG("Requires {} bytes, but {} supplied!\n", sum, weight_str.size());
            ret = 1;    //读入字节数不等也可能可以继续
        }
        ConsoleControl::resetColor();
    }
    auto save_sign = file_bin.get_value("save_sign");
    LOG("Save sign: {}\n", save_sign.substr(0, 100));
    return ret;
}

int Net::weightDataSize() const
{
    int n = 0;
    for (auto& m : weights_)
    {
        if (m)
        {
            n += m->getDataSize();
        }
    }
    return n;
}

float Net::weightSumAbs() const
{
    float l1 = 0;
    for (auto& m : weights_)
    {
        if (m)
        {
            l1 += m->sumAbs();
        }
    }
    return l1;
}

float Net::weightNorm2() const
{
    float l2 = 0;
    for (auto& m : weights_)
    {
        if (m)
        {
            l2 += m->dotSelf();
        }
    }
    return l2;
}

void Net::calNorm(int& n, float& l1, float& l2) const
{
    n = weightDataSize();
    l1 = weightSumAbs();
    l2 = weightNorm2();
}

void Net::outputNorm() const
{
    int n;
    float l1, l2;
    calNorm(n, l1, l2);
    LOG("N = {}, L1 = {}, L2 = {}\n", n, l1, l2);
}

int Net::resetBatchSize(int n)
{
    if (n == getBatch())
    {
        return n;
    }
    //X_.resizeNumber(n);
    getY().resizeNumber(n);
    //A_.resizeNumber(n);
    for (auto& op : op_queue_)
    {
        for (auto& m : op.getMatrixIn())
        {
            if (!m->isWeight())
            {
                m->resizeNumber(n);
            }
        }
        for (auto& m : op.getMatrixOut())
        {
            m->resizeNumber(n);
        }
    }
    for (auto& op : loss_)
    {
        for (auto& m : op.getMatrixIn())
        {
            if (!m->isWeight())
            {
                m->resizeNumber(n);
            }
        }
        for (auto& m : op.getMatrixOut())
        {
            m->resizeNumber(n);
        }
    }
    return n;
}

std::vector<int> Net::getTestGroup()
{
    std::vector<int> group = option_->getVector<int>("train", "test_group");
    if (group.empty())
    {
        if (op_queue_.back().getType() == MatrixOpType::CONCAT)
        {
            for (auto& m : op_queue_.back().getMatrixIn())
            {
                group.push_back(m->getRow());
            }
        }
        else
        {
            group.push_back(op_queue_.back().getMatrixOut()[0]->getRow());
        }
    }
    return group;
}

//测试一个大组
//返回值为0表示基本正常，为负表示输入不合法
//返回值为正表示结果只有0或1，此时有可能是数值有问题
//会改变test_info_和group_test_info_
int Net::test(Matrix* X, Matrix* Y, Matrix* A)
{
    //X不能为空
    if (X == nullptr)
    {
        return -1;
    }
    //Y和A可以有一个为空
    if (Y == nullptr && A == nullptr)
    {
        return -2;
    }
    setDeviceSelf();
    if (weights_.size() <= 0)
    {
        return -1;
    }

    int data_size = X->getNumber();
    if (data_size <= 0)
    {
        return 0;
    }

    gpu_->setActivePhase(ACTIVE_PHASE_TEST);

    float total_error = 0, error = 0;

    //此处注意二者只会有一个为空
    //在执行时，Y可以为空，但是A不能为空
    Matrix temp;
    if (A == nullptr)
    {
        A = &temp;
    }
    if (Y)
    {
        A->resize(Y->getDim());
    }

    auto Xp = getX(), Yp = getY(), Ap = getA();
    int batch = getX().getNumber();
    for (int i = 0; i < data_size; i += batch)
    {
        //检查最后一组是不是组数不足
        int n_rest = std::min(batch, data_size - i);
        if (n_rest < batch)
        {
            //getX().resizeNumber(n_rest);
            //getY().resizeNumber(n_rest);
            //getA().resizeNumber(n_rest);
            resetBatchSize(n_rest);
        }
        if (X->inGpu())
        {
            getX().shareData(*X, 0, i);
        }
        else
        {
            Matrix::copyRows(*X, i, getX(), 0, n_rest);
        }
        if (Y)
        {
            if (Y->inGpu())
            {
                getY().shareData(*Y, 0, i);
            }
            else
            {
                Matrix::copyRows(*Y, i, getY(), 0, n_rest);
            }
        }
        if (A)
        {
            if (A->inGpu())
            {
                getA().shareData(*A, 0, i);
            }
        }

        active(nullptr, nullptr, nullptr, false, &error);

        if (A)
        {
            if (!A->inGpu())
            {
                Matrix::copyRows(getA(), 0, *A, i, n_rest);
            }
        }
        total_error += error * getX().getNumber();
    }
    //还原，如不还原则传入矩阵的显存可能不被释放
    resetBatchSize(batch);
    getX() = Xp;
    getY() = Yp;
    getA() = Ap;
    total_error /= X->getNumber();
    test_info_.error = total_error;

    //恢复网络原来的设置
    gpu_->setActivePhase(ACTIVE_PHASE_TRAIN);

    //若Y为空没有继续测试的必要
    if (Y == nullptr)
    {
        return 0;
    }

    int ret = 0;

    //test_type: 1表示计算每个类别的准确率，一般是分类问题，2表示输出每个样本的结果，一般是图像生成问题

    int test_type = 1;
    if (Y->getWidth() * Y->getHeight() > 1)
    {
        test_type = 2;
    }

    if (test_type == 1)
    {
        int y_size = Y->getRow();
        auto Y_cpu = Y->autoShareClone(UnitType::CPU);
        auto A_cpu = A->autoShareClone(UnitType::CPU);

        ConsoleControl::setColor(CONSOLE_COLOR_RED);
        auto A_max = Matrix(A_cpu.getDim(), A_cpu.getDataType(), UnitType::CPU);

        //查看最后一层是否是拼起来的
        auto group = getTestGroup();
        if (group.size() > 0)
        {
            A_max.fillData(0);
            for (int i_group = 0; i_group < A_cpu.getNumber(); i_group++)    //数据组数
            {
                int total_loc = 0;
                for (int i_combine = 0; i_combine < group.size(); i_combine++)
                {
                    int out = group[i_combine];
                    if (out > 1)
                    {
                        float max_v = -9999;
                        int max_loc = 0;
                        for (int i = 0; i < out; i++)
                        {
                            float v = A_cpu.getData(total_loc + i, i_group);
                            if (v > max_v)
                            {
                                max_v = v;
                                max_loc = i;
                            }
                        }
                        A_max.setData(total_loc + max_loc, i_group, 1);
                    }
                    else if (out == 1)
                    {
                        A_max.setData(total_loc, i_group, A_cpu.getData(total_loc, i_group) > 0.5 ? 1 : 0);
                    }
                    total_loc += out;
                }
            }
        }
        else
        {
            MatrixEx::activeForwardSimple(A_cpu, A_max, ACTIVE_FUNCTION_ABSMAX);
        }
        //A_max->print();

        std::vector<int64_t> right1(y_size, 0), total1(y_size, 0), right0(y_size, 0), total0(y_size, 0);
        std::vector<int64_t> total_group(group.size(), 0), right_group(group.size(), 0);
        std::vector<std::vector<double>> values;    //记录每个类别的输出分布情况，10个桶，分别是0-0.1, 0.1-0.2, ..., 0.9-1.0
        values.resize(y_size);
        for (int i = 0; i < y_size; i++)
        {
            values[i].resize(10);
        }

        for (int i = 0; i < data_size; i++)
        {
            int index = 0;
            for (int i_group = 0; i_group < group.size(); i_group++)
            {
                // 预扫描：判断样本 i 是否属于当前 group（本 group 有 Y=1 才算属于本 group）
                // 若属于其他 group，则本 group 所有输出均为 0，不应计入 total0（否则产生虚假 FP）
                bool group_has_positive = false;
                for (int k = 0; k < group[i_group]; k++)
                {
                    if (Y_cpu.getData(index + k, i) == 1)
                    {
                        group_has_positive = true;
                        break;
                    }
                }

                for (int i_in_group = 0; i_in_group < group[i_group]; i_in_group++)
                {
                    if (Y_cpu.getData(index, i) == 1)
                    {
                        total1[index]++;
                        total_group[i_group]++;
                        if (A_max.getData(index, i) == 1)
                        {
                            right1[index]++;
                            right_group[i_group]++;
                        }
                        auto k = std::min(9, int(floor(A_cpu.getData(index, i) * 10)));
                        if (k >= 0 && k < values[index].size())
                        {
                            values[index][k] += 1;
                        }
                    }
                    if (group_has_positive && Y_cpu.getData(index, i) == 0)
                    {
                        total0[index]++;
                        //total_total++;
                        if (A_max.getData(index, i) == 0)
                        {
                            right0[index]++;
                            //right_total++;
                        }
                    }
                    index++;
                }
            }
        }

        for (int i = 0; i < y_size; i++)
        {
            double sum = std::accumulate(values[i].begin(), values[i].end(), 0.0);
            if (sum > 0)
            {
                for (int j = 0; j < values[i].size(); j++)
                {
                    values[i][j] /= sum;
                }
            }
        }

        //double accuracy_total = 1.0 * right_total / total_total;
        //est_info_.accuracy = accuracy_total;
        //LOG("Total accuracy: {:.2f}% ({}/{}) (error/total)\n", 100 * accuracy_total, total_total - right_total, total_total);

        //int i_group = 0;
        // int total_group = 0;
        std::vector<TestInfo> group_test_info;

        int index = 0;
        for (int i_group = 0; i_group < group.size(); i_group++)
        {
            double accuracy = 1.0 * right_group[i_group] / total_group[i_group];
            LOG("Group {}: {:.2f}% ({}/{}) (error/total)\n", i_group, 100 * accuracy, total_group[i_group] - right_group[i_group], total_group[i_group]);
            // 收集本组各类 TP/FP/FN/TN，用于计算精确率/F1/MCC
            std::vector<int64_t> tp_v, fp_v, fn_v, tn_v;
            for (int i_in_group = 0; i_in_group < group[i_group]; i_in_group++)
            {
                int64_t tp = right1[index], fn = total1[index] - right1[index];
                int64_t tn = right0[index], fp = total0[index] - right0[index];
                tp_v.push_back(tp);
                fp_v.push_back(fp);
                fn_v.push_back(fn);
                tn_v.push_back(tn);
                double accur0, accur1;
                if (group[i_group] == 1)
                {
                    accur0 = 1.0 * right0[index] / total0[index];
                    LOG("Single for two classes\n");
                    LOG("Recall 0: {:.2f}% ({}/{}), ", accur0 * 100.0, total0[index] - right0[index], total0[index]);
                    accur1 = 1.0 * right1[index] / total1[index];
                    LOG("1: {:.2f}% ({}/{})", accur1 * 100.0, total1[index] - right1[index], total1[index]);
                }
                else if (group[i_group] > 1)
                {
                    accur1 = 1.0 * right1[index] / total1[index];
                    LOG("{}: {:.2f}% ({}/{})", i_in_group, accur1 * 100.0, total1[index] - right1[index], total1[index]);
                }
                if (i_in_group < group[i_group] - 1)
                {
                    LOG(", ");
                }
                else
                {
                    LOG("\n");
                }
                //测试信息只记录第一组的，后续组不记录，避免信息过多
                if (i_group == 0)
                {
                    test_info_.accuracy = accuracy;
                    if (group[i_group] == 1)
                    {
                        group_test_info.push_back({ accur0, 0, right0[index], total0[index] });
                    }
                    group_test_info.push_back({ accur1, 0, right1[index], total1[index] });
                }
                index++;
            }
            // 追加精确率、F1分数、MCC（在召回率行之后）
            if (group[i_group] == 1 && !tp_v.empty())
            {
                int64_t tp = tp_v[0], fp = fp_v[0], fn = fn_v[0], tn = tn_v[0];
                double prec1 = (tp + fp) > 0 ? 100.0 * tp / (tp + fp) : 0.0;
                double prec0 = (tn + fn) > 0 ? 100.0 * tn / (tn + fn) : 0.0;
                double f1_1 = (2 * tp + fp + fn) > 0 ? 200.0 * tp / (2 * tp + fp + fn) : 0.0;
                double f1_0 = (2 * tn + fn + fp) > 0 ? 200.0 * tn / (2 * tn + fn + fp) : 0.0;
                double mcc_d = std::sqrt((double)(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
                double mcc = mcc_d > 0 ? (double)(tp * tn - fp * fn) / mcc_d : 0.0;
                LOG("Precision 0: {:.2f}%, 1: {:.2f}%\n", prec0, prec1);
                LOG("F1 score 0: {:.2f}%, 1: {:.2f}%\n", f1_0, f1_1);
                LOG("MCC:      {:.4f}\n", mcc);
            }
            else if (group[i_group] > 1 && !tp_v.empty())
            {
                double macro_f1 = 0.0;
                LOG("Precision");
                for (int k = 0; k < (int)tp_v.size(); k++)
                {
                    double prec = (tp_v[k] + fp_v[k]) > 0 ? 100.0 * tp_v[k] / (tp_v[k] + fp_v[k]) : 0.0;
                    LOG(" {}: {:.2f}%{}", k, prec, k < (int)tp_v.size() - 1 ? "," : "\n");
                }
                LOG("F1 score");
                for (int k = 0; k < (int)tp_v.size(); k++)
                {
                    double f1 = (2 * tp_v[k] + fp_v[k] + fn_v[k]) > 0 ? 200.0 * tp_v[k] / (2 * tp_v[k] + fp_v[k] + fn_v[k]) : 0.0;
                    macro_f1 += f1;
                    LOG(" {}: {:.2f}%{}", k, f1, k < (int)tp_v.size() - 1 ? "," : "\n");
                }
                LOG("Macro F1: {:.2f}%\n", macro_f1 / tp_v.size());
            }
        }

        LOG("Error = {}\n", total_error);
        index = 0;
        for (int i_group = 0; i_group < group.size(); i_group++)
        {
            for (int i_in_group = 0; i_in_group < group[i_group]; i_in_group++)
            {
                LOG("{}-{}: {::.3f}\n", i_group, i_in_group, values[index]);
                index++;
                //if (group.size() > 0 && i == group[0] - 1)
                //{
                //break;
                //}
            }
        }
        group_test_info_ = std::move(group_test_info);
        ConsoleControl::resetColor();
    }
    else if (test_type == 2)
    {
        auto Y_cpu = Y->createSharedCol(0, 10).autoShareClone(UnitType::CPU);
        auto A_cpu = A->createSharedCol(0, 10).autoShareClone(UnitType::CPU);

        LOG("Error = {}\n", total_error);

        // 分割语义评估指标（对全部测试样本）
        {
            auto Y_all = Y->autoShareClone(UnitType::CPU);
            auto A_all = A->autoShareClone(UnitType::CPU);
            int nc = Y->getChannel();
            int W = Y->getWidth(), H = Y->getHeight(), N = Y->getNumber();
            double mean_iou = 0, mean_dice = 0;
            for (int ic = 0; ic < nc; ic++)
            {
                int64_t TP = 0, FP = 0, FN = 0, TN = 0;
                for (int n = 0; n < N; n++)
                {
                    for (int ih = 0; ih < H; ih++)
                    {
                        for (int iw = 0; iw < W; iw++)
                        {
                            int gt = Y_all.getData(iw, ih, ic, n) > 0.5f ? 1 : 0;
                            int pr = A_all.getData(iw, ih, ic, n) > 0.5f ? 1 : 0;
                            if (gt == 1 && pr == 1) { TP++; }
                            else if (gt == 0 && pr == 1) { FP++; }
                            else if (gt == 1 && pr == 0) { FN++; }
                            else { TN++; }
                        }
                    }
                }
                double iou = (TP + FP + FN > 0) ? (double)TP / (TP + FP + FN) : 1.0;
                double dice = (2 * TP + FP + FN > 0) ? (double)(2 * TP) / (2 * TP + FP + FN) : 1.0;
                double prec = (TP + FP > 0) ? (double)TP / (TP + FP) : 1.0;
                double rec = (TP + FN > 0) ? (double)TP / (TP + FN) : 1.0;
                double pacc = (TP + TN + FP + FN > 0) ? (double)(TP + TN) / (TP + TN + FP + FN) : 1.0;
                LOG("Ch{}: IoU={:.4f}, Dice={:.4f}, Prec={:.4f}, Rec={:.4f}, PixAcc={:.4f}\n",
                    ic, iou, dice, prec, rec, pacc);
                mean_iou += iou;
                mean_dice += dice;
            }
            if (nc > 0)
            {
                mean_iou /= nc;
                mean_dice /= nc;
                if (nc > 1)
                {
                    LOG("mIoU={:.4f}, mDice={:.4f}\n", mean_iou, mean_dice);
                }
                test_info_.accuracy = mean_iou;
            }
        }

        LOG("Label --> Infer\n");
        //std::string chars = R"( `.^,:~"<!ct+{i7?u30pw4A8DX%#HWM)";
        std::string chars = " .oO";
        auto out_char = [&chars](float f)
        {
            const int m = chars.size();
            int n = f * m;
            n = std::max(0, std::min(m - 1, n));
            return chars[n];
        };
        auto out_matrix = [&out_char](Matrix& M1, Matrix& M2, int n)
        {
            int step = std::max(1, M1.getHeight() / 32);
            for (int ih = 0; ih < M1.getHeight(); ih += step)
            {
                LOG("[");
                for (int iw = 0; iw < M1.getWidth(); iw += step)
                {
                    LOG("{}", out_char(M1.getData(iw, ih, 0, n)));
                }
                LOG("]         [");
                for (int iw = 0; iw < M2.getWidth(); iw += step)
                {
                    LOG("{}", out_char(M2.getData(iw, ih, 0, n)));
                }
                LOG("]\n");
            }
        };
        out_matrix(Y_cpu, A_cpu, 0);
    }
    test2(&getX(), &getY(), &getA());    //供子类扩展测试内容，直接引用的net内部的矩阵
    return ret;
}

//将所有参数集中在同一块内存，方便并行中的数据交换
void Net::combineWeights(std::vector<Matrix*>& weights, Matrix& result)
{
    setDeviceSelf();

    //需对齐显存，否则速度下降会比较严重
    auto c256 = [](int64_t i)
    {
        return (i + 255) / 256 * 256;
    };

    int64_t sum = 0;
    for (int i = 0; i < weights.size(); i++)
    {
        auto m = weights[i];
        sum += c256(m->getDataSize());
    }
    result.resize(1, sum);
    auto& dparameters = result.d();
    result.fillData(0);
    dparameters.fillData(0);
    int64_t p = 0;
    int mode = 2;
    for (int i = 0; i < weights.size(); i++)
    {
        auto& m = weights[i];
        //只合并需要训练的参数
        //并行计算时，只有需要训练的参数才会被传递
        if (m && m->needBack())
        {
            if (mode == 0 || mode == 2)
            {
                auto m1 = m->clone();
                m->shareData(result, 0, p);
                Matrix::copyData(m1, *m);
            }
            if (mode == 1 || mode == 2)
            {
                m->d().shareData(dparameters, 0, p);
            }
            p += c256(m->getDataSize());
            //LOG("combined parameter size = %lld\n", p);
        }
    }
}

void Net::initWeights()
{
    auto filler = option_->getEnum("train", "init_weight", RANDOM_FILL_XAVIER);
    // 推理模式下（train_epochs=0）权重将从文件加载，跳过随机初始化以节省内存和避免 CUDA 状态问题
    bool skip_fill = (option_->getInt("train", "train_epochs", 0) == 0);
    //LOG("Initialized weight method is {}\n", option_->getStringFromEnum(filler));
    for (auto& op : op_queue_)
    {
        for (auto& m : op.getMatrixIn())
        {
            //与X无链接，与Loss有链接，即属于权重
            if (m->isWeight())
            {
                if (!VectorMath::vector_have(weights_, m.get()))
                {
                    if (!m->needLoad())
                    {
                        weights_.push_back(m.get());
                        continue;
                    }
                    int one_channel = m->getRow() / m->getChannel();
                    auto in = m->getChannel() * one_channel;
                    auto out = m->getNumber() * one_channel;
                    if (filler != RANDOM_FILL_XAVIER)
                    {
                        if (op.getMatrixIn().size() > 0)
                        {
                            for (auto& mi : op.getMatrixIn())
                            {
                                if (mi.get() != m.get())
                                {
                                    in = mi->getRow();
                                    break;
                                }
                            }
                        }
                        if (op.getMatrixOut().size() > 0)
                        {
                            out = op.getMatrixOut()[0]->getRow();
                        }
                    }
                    MatrixEx::fill(*m, skip_fill ? RANDOM_FILL_CONSTANT : filler, in, out);
                    if (op.getType() == MatrixOpType::ADD_BIAS)
                    {
                        m->fillData(0);
                    }
                    //LayerNorm: scale 初始化为 1, bias 初始化为 0
                    if (op.getType() == MatrixOpType::LAYER_NORM)
                    {
                        auto& mats = op.getMatrixIn();
                        if (mats.size() >= 3 && m.get() == mats[1].get())
                        {
                            m->fillData(1);
                        }
                        else if (mats.size() >= 3 && m.get() == mats[2].get())
                        {
                            m->fillData(0);
                        }
                    }
                    //RMSNorm: scale 初始化为 1
                    if (op.getType() == MatrixOpType::RMS_NORM)
                    {
                        auto& mats = op.getMatrixIn();
                        if (mats.size() >= 2 && m.get() == mats[1].get())
                        {
                            m->fillData(1);
                        }
                    }
                    //m->scale(10);    //调试用
                    weights_.push_back(m.get());
                }
            }
        }
        //检查没有权重的情况
        if (op.getMatrixIn().size() > 1)
        {
            int weight = 0;
            for (auto& m : op.getMatrixIn())
            {
                if (m->isWeight())
                {
                    weight++;
                }
            }
            if (weight == 0)
            {
                //一般来说，残差相加，连接，注意力加权没有权重是正常的
                //LOG("Warning: no weight in op {}!\n", MatrixOp::getOpName(op.getType()));
            }
        }
    }
    // 仅训练时才需要将权重合并到连续缓冲（用于 solver 梯度更新），推理跳过以节省显存
    if (option_->getInt("train", "train_epochs", 0) > 0)
    {
        combineWeights(weights_, all_weights_);
    }
}

std::string Net::ir()
{
    std::string ir;
    //for (auto& w : weights_)
    //{
    //    ir += std::format("M{} = {};", (uint64_t)w, w->sizeMessage());
    //}
    X_->setIsInput(false);
    ir += std::format("{} = {};", X_, X_->sizeMessage(0));
    ir += MatrixOp::inference_ir(op_queue_);
    ir += MatrixOp::inference_ir(loss_);
    ir += std::format("setXY({}, {});", X_, A_);
    X_->setIsInput(true);
    return ir;
}

//利用已知网络修改X适应答案，只处理一个minibatch
void Net::attack(Matrix* X, Matrix* Y)
{
    //active(X, Y, nullptr, false, nullptr);

    //for (int i_layer = getLayersCount() - 1; i_layer >= 0; i_layer--)
    //{
    //    layer_vector_[i_layer]->activeBackward();
    //}
    //getFirstLayer()->updateABackward();
}

void Net::clearTime()
{
    for (auto& op : op_queue_)
    {
        op.clearTime();
    }
}

void Net::outputTime() const
{
    for (auto& op : op_queue_)
    {
        std::string active_str;
        if (op.getActiveType() != ACTIVE_FUNCTION_NONE)
        {
            active_str = "-" + option_->getStringFromEnum(op.getActiveType());
        }
        LOG("({}) {}{}: {}, {}\n", op.getIndex(), MatrixOp::getOpName(op.getType()), active_str, op.getForwardTime(), op.getBackwardTime());
    }
    for (auto& op : loss_)
    {
        LOG("(loss) {}: {}, {}\n", MatrixOp::getOpName(op.getType()), op.getForwardTime(), op.getBackwardTime());
    }
}

void Net::inference(Matrix& X, Matrix& A)    //仅计算，不提供前后处理
{
    setDeviceSelf();
    auto Xp = getX(), Yp = getY(), Ap = getA();
    int batch = getX().getNumber();
    int data_size = X.getNumber();
    for (int i = 0; i < data_size; i += batch)
    {
        //检查最后一组是不是组数不fut
        int n_rest = std::min(batch, data_size - i);
        if (n_rest < batch)
        {
            //getX().resizeNumber(n_rest);
            //getY().resizeNumber(n_rest);
            //getA().resizeNumber(n_rest);
            resetBatchSize(n_rest);
        }
        if (X.inGpu())
        {
            getX().shareData(X, 0, i);
        }
        else
        {
            Matrix::copyRows(X, i, getX(), 0, n_rest);
        }

        if (A.inGpu())
        {
            getA().shareData(A, 0, i);
        }

        active(nullptr, nullptr, nullptr, false, nullptr);

        if (!A.inGpu())
        {
            Matrix::copyRows(getA(), 0, A, i, n_rest);
        }
        //LOG("{}", i);
    }
    //还原，如不还原则传入矩阵的显存可能不被释放
    resetBatchSize(batch);
    getX() = Xp;
    getY() = Yp;
    getA() = Ap;
}

void Net::optimizeActivationMemory()
{
    // 调试开关：允许通过环境变量/配置关闭激活复用，避免输出张量被复用池覆盖。
    // ini: [train] disable_activation_memory_reuse=1
    if (option_->getInt("train", "disable_activation_memory_reuse", 0) != 0)
    {
        LOG("Activation memory reuse disabled\n");
        return;
    }

    // 仅推理模式（train_epochs == 0）才安全复用激活张量
    if (option_->getInt("train", "train_epochs", 0) > 0)
    {
        return;
    }

    // 诊断：VAE 是否有计算图
    if (op_queue_.empty())
    {
        LOG("ActMemReuse: empty op_queue (no computation graph, possibly weights-only network)\n");
        return;
    }

    const bool reuse_debug = (option_->getInt("train", "activation_reuse_debug", 0) != 0);

    // Step 1: 构建受保护张量集合（不参与复用）
    // 先收集图内张量，避免把 in-graph 的 extra_matrix 过度保护，
    // 让它们回到生命周期分析路径。
    std::unordered_set<Matrix*> in_graph_set;
    // 记录在 op_queue_ 中被写入（作为 MatrixOut）的张量。
    // extra_matrixsp_ 里"在图中但从未被写入"的张量是只读外部输入
    //（如 text embedding、timestep embedding 等），它们可能在多步推理中复用，
    // 必须保护，否则复用池会覆盖其内容，导致后续步骤使用错误的条件嵌入。
    std::unordered_set<Matrix*> written_in_graph_set;
    for (auto& op : op_queue_)
    {
        for (auto& m : op.getMatrixIn())
        {
            if (m)
            {
                in_graph_set.insert(m.get());
            }
        }
        for (auto& m : op.getMatrixOut())
        {
            if (m)
            {
                in_graph_set.insert(m.get());
                written_in_graph_set.insert(m.get());
            }
        }
    }

    std::unordered_set<Matrix*> protected_set;
    if (X_)
    {
        protected_set.insert(X_.get());
    }
    if (Y_)
    {
        protected_set.insert(Y_.get());
    }
    if (A_)
    {
        protected_set.insert(A_.get());
    }
    for (auto& [name, m] : extra_matrixsp_)
    {
        if (!m)
        {
            continue;
        }
        const bool in_graph = in_graph_set.count(m.get()) > 0;
        // 保护"跨网络持久输入"：data_ 已预分配（来自另一个 Net 的输出，如 text embedding）
        // 且在本图中从不写入。这类张量在多步推理（如 UNet 的 T 次去噪）中每步都被读取，
        // 一旦被复用池覆盖，后续步骤将使用错误的条件嵌入，导致图像完全错误。
        // 注意：仅检查 data_!=null（预分配），不保护懒惰分配（data_=null）张量；
        // 后者是本 Net 内部的中间激活，可以安全参与复用（如 cap embedder 内的中间激活）。
        const bool never_written = !written_in_graph_set.count(m.get());
        if (!in_graph || m->isWeight() || never_written)
        {
            protected_set.insert(m.get());
        }
    }
    for (auto& op : op_queue_)
    {
        for (auto& m : op.getMatrixIn())
        {
            if (m && m->isWeight())
            {
                protected_set.insert(m.get());
            }
        }
    }

    // Step 2: 构建 view/alias 集合（shareData 自上游的张量，不能在池中单独归还）
    // 只按 Matrix* 做活性分析会漏掉同一存储上的别名，导致提前回收并覆写，
    // 在 VAE 这类深层网络中会放大成噪声输出。
    std::unordered_set<Matrix*> view_set;
    std::unordered_map<const void*, std::unordered_set<Matrix*>> storage_groups;
    auto collect_storage = [&](const MatrixSP& m)
    {
        if (!m)
        {
            return;
        }
        storage_groups[m->getStorageKey()].insert(m.get());
    };
    collect_storage(X_);
    collect_storage(Y_);
    collect_storage(A_);
    for (auto& [name, m] : extra_matrixsp_)
    {
        collect_storage(m);
    }
    for (auto& op : op_queue_)
    {
        for (auto& m : op.getMatrixIn())
        {
            collect_storage(m);
        }
        for (auto& m : op.getMatrixOut())
        {
            collect_storage(m);
        }
        if (op.getType() == MatrixOpType::RESHAPE || op.getType() == MatrixOpType::KV_CACHE)
        {
            for (auto& m : op.getMatrixOut())
            {
                if (m)
                {
                    view_set.insert(m.get());
                }
            }
        }
    }

    // 所有共享同一底层存储的张量都视作别名，参与统一活性期。
    // 这些张量不直接加入复用池，避免提前回收覆盖仍在使用的数据。
    std::unordered_set<Matrix*> alias_set;
    for (auto& [key, mats] : storage_groups)
    {
        if (mats.size() > 1)
        {
            for (auto* m : mats)
            {
                alias_set.insert(m);
            }
        }
    }

    // Step 3: 计算每个张量最后一次被读取的 op 序号（last_read）
    std::unordered_map<Matrix*, int> last_read;
    std::unordered_map<Matrix*, int> first_write;
    for (int i = 0; i < (int)op_queue_.size(); i++)
    {
        for (auto& m : op_queue_[i].getMatrixOut())
        {
            if (m && !first_write.count(m.get()))
            {
                first_write[m.get()] = i;
            }
        }
        for (auto& m : op_queue_[i].getMatrixIn())
        {
            if (m)
            {
                last_read[m.get()] = i;
            }
        }
    }
    // 同存储别名统一 last_read，避免 source/view 生命周期被拆开导致过早复用。
    for (auto& [key, mats] : storage_groups)
    {
        int max_lr = -1;
        for (auto* m : mats)
        {
            auto it = last_read.find(m);
            if (it != last_read.end())
            {
                max_lr = std::max(max_lr, it->second);
            }
        }
        if (max_lr >= 0)
        {
            for (auto* m : mats)
            {
                last_read[m] = max_lr;
            }
        }
    }
    // 将 RESHAPE view 的 last_read 传播到其源头（source tensor），
    // 防止源头缓冲在 view 仍在使用时被复用
    for (auto& op : op_queue_)
    {
        if (op.getType() == MatrixOpType::RESHAPE)
        {
            if (op.getMatrixIn().empty() || op.getMatrixOut().empty())
            {
                continue;
            }
            auto& src = op.getMatrixIn()[0];
            auto& view = op.getMatrixOut()[0];
            if (!src || !view)
            {
                continue;
            }
            auto it_view = last_read.find(view.get());
            if (it_view != last_read.end())
            {
                auto& lr_src = last_read[src.get()];
                lr_src = std::max(lr_src, it_view->second);
            }
        }
    }

    if (reuse_debug)
    {
        int protected_cnt = 0, view_cnt = 0, alias_cnt = 0, candidate_cnt = 0;
        int printed = 0;
        std::unordered_set<Matrix*> seen;
        auto dump_one = [&](const MatrixSP& m)
        {
            if (!m || !seen.insert(m.get()).second)
            {
                return;
            }
            const bool is_protected = protected_set.count(m.get()) > 0;
            const bool is_view = view_set.count(m.get()) > 0;
            const bool is_alias = alias_set.count(m.get()) > 0;
            const bool candidate = !is_protected && !is_view && !is_alias && m->inGpu() && m->getDataSize() > 0;
            protected_cnt += is_protected ? 1 : 0;
            view_cnt += is_view ? 1 : 0;
            alias_cnt += is_alias ? 1 : 0;
            candidate_cnt += candidate ? 1 : 0;

            if (printed < 24)
            {
                auto fw_it = first_write.find(m.get());
                auto lr_it = last_read.find(m.get());
                LOG("ActMemDiag: ptr={} size={}KB dt={} fw={} lr={} flags[P{} V{} A{} C{}] name='{}'\n",
                    (void*)m.get(),
                    m->getDataSizeInByte() / 1024,
                    m->getDataTypeByInt(),
                    fw_it == first_write.end() ? -1 : fw_it->second,
                    lr_it == last_read.end() ? -1 : lr_it->second,
                    is_protected ? 1 : 0,
                    is_view ? 1 : 0,
                    is_alias ? 1 : 0,
                    candidate ? 1 : 0,
                    m->getWeightName());
                printed++;
            }
        };
        for (auto& op : op_queue_)
        {
            for (auto& m : op.getMatrixIn())
            {
                dump_one(m);
            }
            for (auto& m : op.getMatrixOut())
            {
                dump_one(m);
            }
        }
        LOG("ActMemDiag: protected={} view={} alias={} candidate={} (printed={})\n",
            protected_cnt, view_cnt, alias_cnt, candidate_cnt, printed);
    }

    // Step 4: 贪心显存池分配（best-fit）+ 立即分配 GPU 显存
    // 在同一次遍历中完成"规划复用"与"分配显存"，避免两阶段之间指针状态不一致。
    // 规则：
    //   · can_reuse=true，池中有合适缓冲 → shareData + 立即 fillData(0)
    //   · can_reuse=true，池中无合适缓冲 → resize(false,false) 分配新显存
    //   · can_reuse=false（受保护/view/alias）→ resize(false,false) 同步或分配
    //   · 已分配（data_ptr!=null）→ 跳过（权重、预分配外部张量等）
    std::unordered_map<int, std::map<int64_t, std::vector<MatrixSP>>> pool;
    int64_t total_saved_bytes = 0;
    int reuse_count = 0;
    const bool reuse_zero_fill = (option_->getInt("train", "activation_reuse_zero_fill", 1) != 0);
    int oversized_reuse_count = 0;

    auto can_reuse = [&](const MatrixSP& m)
    {
        if (!m)
        {
            return false;
        }
        if (protected_set.count(m.get()) || view_set.count(m.get()) || alias_set.count(m.get()))
        {
            return false;
        }
        if (!m->inGpu() || m->getDataSize() == 0)
        {
            return false;
        }
        return true;
    };
    auto acquire_from_pool = [&](const MatrixSP& m, const std::unordered_set<void*>& forbidden_ptrs) -> MatrixSP
    {
        auto& by_size = pool[m->getDataTypeByInt()];
        std::vector<MatrixSP> skipped;
        for (auto it = by_size.lower_bound(m->getDataSizeInByte()); it != by_size.end();)
        {
            auto& entries = it->second;
            while (!entries.empty())
            {
                MatrixSP r = entries.back();
                entries.pop_back();
                if (!forbidden_ptrs.count(r->getDataPtr()))
                {
                    for (auto& s : skipped)
                    {
                        by_size[s->getDataSizeInByte()].push_back(s);
                    }
                    if (entries.empty())
                    {
                        by_size.erase(it);
                    }
                    return r;
                }
                skipped.push_back(r);
            }
            it = by_size.erase(it);
        }
        for (auto& s : skipped)
        {
            by_size[s->getDataSizeInByte()].push_back(s);
        }
        return nullptr;
    };
    auto release_to_pool = [&](const MatrixSP& m)
    {
        pool[m->getDataTypeByInt()][m->getDataSizeInByte()].push_back(m);
    };

    int candidate_total = 0;
    int candidate_with_last_read = 0;
    {
        std::unordered_set<Matrix*> seen;
        auto count_one = [&](const MatrixSP& m)
        {
            if (!m || !seen.insert(m.get()).second)
            {
                return;
            }
            if (!can_reuse(m))
            {
                return;
            }
            candidate_total++;
            if (last_read.count(m.get()))
            {
                candidate_with_last_read++;
            }
        };
        for (auto& op : op_queue_)
        {
            for (auto& m : op.getMatrixIn())
            {
                count_one(m);
            }
            for (auto& m : op.getMatrixOut())
            {
                count_one(m);
            }
        }
    }

    // 预释放阶段：将全部候选激活张量的显存统一归还系统。
    // 确保贪心阶段在懒分配（data_=null，releaseForReuse 为空操作）和非懒分配
    // （data_!=null，实际释放显存）两种模式下都能从零开始规划复用，
    // 而不依赖 data_ 的初始状态。
    {
        std::unordered_set<Matrix*> seen_pr;
        for (auto& op : op_queue_)
        {
            for (auto& m : op.getMatrixOut())
            {
                if (!m || !seen_pr.insert(m.get()).second) { continue; }
                if (can_reuse(m)) { m->releaseForReuse(); }
            }
        }
    }

    LOG("ActMem: greedy alloc start (ops={} candidates={})\n", (int)op_queue_.size(), candidate_total);

    for (int i = 0; i < (int)op_queue_.size(); i++)
    {
        auto& op = op_queue_[i];
        std::unordered_set<void*> input_ptrs;
        for (auto& in : op.getMatrixIn())
        {
            if (in && in->getDataPtr() != nullptr)
            {
                input_ptrs.insert(in->getDataPtr());
            }
        }

        // 为本 op 的输出张量分配/复用显存
        for (auto& m : op.getMatrixOut())
        {
            if (!m || m->getDataSize() <= 0)
            {
                continue;
            }
            if (m->getDataPtr() != nullptr)
            {
                continue;    // 已分配（权重、预分配张量）
            }

            if (can_reuse(m))
            {
                MatrixSP pooled = acquire_from_pool(m, input_ptrs);
                if (pooled)
                {
                    // pooled 已持有真实 GPU 指针；shareData 后 m 立刻拥有有效指针，
                    // fillData(0) 可立即执行，无需任何延迟。
                    const int64_t req_sz = m->getDataSizeInByte();
                    const int64_t got_sz = pooled->getDataSizeInByte();
                    m->shareData(*pooled);
                    if (reuse_zero_fill)
                    {
                        // 清零复用缓冲，防止上一轮残留数据在部分覆写的 kernel 中产生伪影
                        if (m->getDataPtr() == nullptr)
                        {
                            LOG("ActMem: WARN fillData on null ptr at op {} m.size={}KB\n", i, m->getDataSizeInByte() / 1024);
                        }
                        else
                        {
                            m->fillData(0);
                        }
                    }
                    total_saved_bytes += req_sz;
                    reuse_count++;
                    if (got_sz > req_sz)
                    {
                        oversized_reuse_count++;
                    }
                }
                else
                {
                    // 无可用缓冲，分配新显存
                    m->resize(m->getDim(), false, false);
                }
            }
            else
            {
                // 受保护/view/alias 张量：同步指针（view）或分配新显存（其余）
                m->resize(m->getDim(), false, false);
            }
        }

        // 将本 op 中最后一次使用（last_read == i）的输入张量归还给池
        std::unordered_set<Matrix*> released_now;
        for (auto& m : op.getMatrixIn())
        {
            if (!can_reuse(m))
            {
                continue;
            }
            auto it = last_read.find(m.get());
            if (it != last_read.end() && it->second == i)
            {
                release_to_pool(m);
                released_now.insert(m.get());
            }
        }

        // 对于从未被读取的临时输出，写完当前 op 后即可归还。
        for (auto& m : op.getMatrixOut())
        {
            if (!can_reuse(m) || released_now.count(m.get()))
            {
                continue;
            }
            if (!last_read.count(m.get()))
            {
                release_to_pool(m);
            }
        }
    }

    LOG("ActMem: greedy alloc done (reused={} freed={:.1f}MB)\n", reuse_count, total_saved_bytes / 1048576.0);
    // 补全：分配在 op_queue_ 主循环中未覆盖到的张量
    // （X_/Y_/A_、loss 张量、extra_matrixsp_ 中不在图输出路径上的项等）
    auto ensure_alloc = [](const MatrixSP& m)
    {
        if (!m || m->getDataSize() <= 0 || m->getDataPtr() != nullptr)
        {
            return;
        }
        m->resize(m->getDim(), false, false);
    };
    ensure_alloc(X_);
    ensure_alloc(Y_);
    ensure_alloc(A_);
    for (auto& [name, m] : extra_matrixsp_)
    {
        ensure_alloc(m);
    }
    for (auto& op : op_queue_)
    {
        for (auto& m : op.getMatrixIn())
        {
            ensure_alloc(m);
        }
        for (auto& m : op.getMatrixOut())
        {
            ensure_alloc(m);
        }
    }
    for (auto& op : loss_)
    {
        for (auto& m : op.getMatrixIn())
        {
            ensure_alloc(m);
        }
        for (auto& m : op.getMatrixOut())
        {
            ensure_alloc(m);
        }
    }

    LOG("ActMem: ensure_alloc done\n");
    LOG("ActMemReuse: freed={:.1f}MB reused={} oversized={} candidate={} protected={} view={} alias={}\n",
        total_saved_bytes / 1048576.0,
        reuse_count,
        oversized_reuse_count,
        candidate_total,
        (int)protected_set.size(),
        (int)view_set.size(),
        (int)alias_set.size());
}

void Net::buildForwardGraph()
{
#if !ENABLE_CUDA
    use_graph_ = false;
    LOG("[Net] use_cudnn_graph requested but build has no CUDA support; ignored.\n");
    return;
#else
    if (!getX().isCuda())
    {
        LOG("[Net] use_cudnn_graph requested but X is not on CUDA; falling back to plain.\n");
        use_graph_ = false;
        return;
    }
    fwd_graph_ = std::make_unique<CudnnOpQueueGraph>();
    if (!fwd_graph_->build(gpu_->cudnn_handle_, gpu_, op_queue_))
    {
        fwd_graph_.reset();
        LOG("[Net] whole-network cuDNN graph build failed; using plain per-op path.\n");
    }
    else
    {
        LOG("[Net] whole-network cuDNN graph built (ops={}).\n", (int)op_queue_.size());
    }
#endif
}

void Net::runForwardWithGraph()
{
#if ENABLE_CUDA
    int rc = fwd_graph_->execute();
    if (rc != 0)
    {
        LOG("[Net] cuDNN graph execute failed (rc={}), falling back to plain.\n", rc);
        fwd_graph_.reset();
        MatrixOp::forward(op_queue_);
    }
#else
    MatrixOp::forward(op_queue_);
#endif
}

}    // namespace cccc