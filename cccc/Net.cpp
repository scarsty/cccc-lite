#include "Net.h"
#include "ConsoleControl.h"
#include "INIReaderBin.h"
#include "MatrixEx.h"
#include "Timer.h"
#include "VectorMath.h"
#include "filefunc.h"

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
    int r = init2();
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

    solver_.init(option_, "train", all_weights_);

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
    LOG("{}\n", ir());
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
    MatrixOp::forward(op_queue_);
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
    if (filefunc::fileExist(str))
    {
        buffer = filefunc::readFileToString(str);
    }
    else if (load_mode)
    {
        buffer = str;
    }

    if (buffer.size() <= 0)
    {
        LOG("failed!\n");
        return -2;
    }

    int ret = 0;
    INIReaderBin file_bin;
    file_bin.parse(buffer);
    auto data_type = option_->getEnumFromString<DataType>(file_bin.get_value("data_type"));
    LOG("Data type of save file is {}\n", option_->getStringFromEnum(data_type));
    LOG("Data type of net is {}\n", option_->getStringFromEnum(weights_[0]->getDataType()));
    if (file_bin.has_value("weight_bin0"))
    {
        //分开保存的情况
        int index = 0;
        for (auto& m : weights_)
        {
            if (m->needLoad())
            {
                std::string key = "weight_bin" + std::to_string(index);
                auto weight_str = file_bin.get_value(key);
                if (data_type == m->getDataType())
                {
                    m->load((char*)weight_str.data(), weight_str.size());
                }
                else
                {
                    Matrix m1(m->getDim(), m->getDataType(), UnitType::CPU);
                    for (int64_t i = 0; i < m->getDataSize(); i++)
                    {
                        m1.setData(i, weight_str.data(), data_type);
                    }
                    Matrix::copyData(m1, *m);
                }
                if (weight_str.size() / MatrixData::getDataTypeSize(data_type) != m->getDataSize())
                {
                    ConsoleControl::setColor(CONSOLE_COLOR_RED);
                    LOG("Warning: weight {} requires {} bytes, but {} supplied!\n", index, m->getDataSizeInByte(), weight_str.size());
                    ConsoleControl::resetColor();
                    ret = 1;    //读入字节数不等也可能可以继续
                }
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
                    continue;
                }
                if (data_type == m->getDataType())
                {
                    sum += m->load((char*)weight_str.data() + sum, (weight_str.size() - sum) / m->getDataTypeSize());
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
                    if (Y_cpu.getData(index, i) == 0)
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
            for (int i_in_group = 0; i_in_group < group[i_group]; i_in_group++)
            {
                double accur0, accur1;
                if (group[i_group] == 1)
                {
                    accur0 = 1.0 * right0[index] / total0[index];
                    LOG("Single for two classes\n");
                    LOG("0: {:.2f}% ({}/{}), ", accur0 * 100.0, total0[index] - right0[index], total0[index]);
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
                    MatrixEx::fill(*m, filler, in, out);
                    if (op.getType() == MatrixOpType::ADD_BIAS)
                    {
                        m->fillData(0);
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
    combineWeights(weights_, all_weights_);
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

void Net::test_only(Matrix& X, Matrix& A)    //仅计算，不提供前后处理
{
    setDeviceSelf();
    auto Xp = getX(), Yp = getY(), Ap = getA();
    int batch = getX().getNumber();
    int data_size = X.getNumber();
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

}    // namespace cccc