#include "Net.h"
#include "ConsoleControl.h"
#include "INIReaderBin.h"
#include "Layer.h"
#include "Timer.h"
#include "VectorMath.h"
#include "filefunc.h"
#include "strfunc.h"
#include <algorithm>

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
    //MatrixOp::simpleQueue(op_queue_, getX(), getA());
    if (op_queue_.size() == 0)
    {
        LOG("Empty compute queue!\n");
        r = -2;
    }
    getX().setNeedReverse(false);
    initWeights();
    solver_.init(option_, "train", all_weights_);
    //workspace_.resize(all_weights_);

    if (loss_.empty())    //未设置损失函数就自己添加一个
    {
        MatrixOp op;
        op.set(MatrixOpType::LOSS, { A_, Y_, loss_weight_ }, {}, {}, {}, {});
        loss_.push_back(op);
        //LossWeight_->printAsMatrix();
    }

#if defined(_DEBUG) || !defined(_WIN32)
    LOG("{}\n", ir());
    //MatrixOp::ir(loss_);
#endif
    if (getBatch() <= 0)
    {
        resetBatchSize(1);
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

    seperate_update_weight_ = option_->getInt("train", "seperate_update_weight", 0);

    return r;
}

//learn为真时，会反向更新网络
//active只处理一个gpu中的minibatch
//A是外部提供的矩阵，用于保存结果
void Net::active(Matrix* X, Matrix* Y, Matrix* A, bool back, realc* error)
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
    MatrixOp::forward(op_queue_);
    if (back)
    {
        MatrixOp::backward(op_queue_, loss_, true);
    }
    if (A)
    {
        Matrix::copyDataPtr(getA(), getA().getDataPtr(), *A, A->getDataPtr(), getA().getDataSize());
    }
    if (error && getY().getNumber() > 0)
    {
        Matrix R(getA().getDim());
        Matrix::add(getA(), getY(), R, 1, -1);
        *error = R.dotSelf() / getY().getNumber();
    }
}

void Net::updateWeight()
{
    if (seperate_update_weight_)
    {
        for (auto& w : weights_)
        {
            solver_.updateWeights(*w, getX().getNumber());
        }
    }
    else
    {
        solver_.updateWeights(all_weights_, getX().getNumber());
    }
}

//保存权重，需配合ini中的网络结构
//返回值：0正常，其他值不正常
int Net::saveWeight(const std::string& filename, const std::string& sign)
{
    setDeviceSelf();
    if (filename == "")
    {
        return -1;
    }
    LOG("Save net to {}... ", filename);
    filefunc::makePath(filefunc::getParentPath(filename));

    int sum = 0;
    for (auto& m : weights_)
    {
        sum += m->getDataSize();
    }
    std::string buffer(sum * sizeof(real), '\0');
    auto p = (real*)buffer.data();
    for (auto& m : weights_)
    {
        p += m->save(p, m->getDataSize());
    }

    INIReaderBin file_bin;
    file_bin.set_value("weight_binary", buffer);
    if (!sign.empty())
    {
        file_bin.set_value("save_sign", sign);
    }

    if (file_bin.save(filename) > 0)
    {
        LOG("done\n");
        LOG("Save sign: {}\n", sign.c_str());
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
//1    缓冲区尺寸大于网络尺寸，即网络的所有参数都被赋值，但缓冲区内仍有数据，此时通常不会是正常结果
//-1   读入长度小于网络尺寸，即网络的部分参数并未被赋值，此时通常不会是正常结果
//-2   不能读入文件
int Net::loadWeight(const std::string& str, int load_mode)
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
        buffer = strfunc::readStringFromFile(str);
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

    INIReaderBin file_bin;
    file_bin.parse(buffer);
    auto weight_str = file_bin.get_value("weight_binary");
    if (weight_str.empty())
    {
        weight_str = buffer;    //直接当成二进制流也可
    }
    int sum = 0;
    for (auto& m : weights_)
    {
        sum += m->load((real*)weight_str.data() + sum, (buffer.size() - sum * sizeof(real)) / sizeof(real));
    }
    LOG("done\n");
    int ret = 0;
    ConsoleControl::setColor(CONSOLE_COLOR_RED);
    if (weight_str.size() > sum * sizeof(real))
    {
        LOG("Warning: size of weight is longer than net!\n");
        ret = 1;
    }
    else if (weight_str.size() < sum * sizeof(real))
    {
        LOG("Warning: size of weight is shorter than net!\n");
        ret = 1;
    }
    ConsoleControl::setColor(CONSOLE_COLOR_NONE);

    std::string sign = file_bin.get_value("save_sign");
    if (sign.empty())
    {
        ConsoleControl::setColor(CONSOLE_COLOR_RED);
        LOG("Warning: no save sign!\n");
        ConsoleControl::setColor(CONSOLE_COLOR_NONE);
        //ret = 2;
    }
    else
    {
        if (sign.size() > 100)
        {
            sign = sign.substr(0, 100);
        }
        LOG("Save sign: {}\n", sign);
    }
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

realc Net::weightSumAbs() const
{
    realc l1 = 0;
    for (auto& m : weights_)
    {
        if (m)
        {
            l1 += m->sumAbs();
        }
    }
    return l1;
}

realc Net::weightNorm2() const
{
    realc l2 = 0;
    for (auto& m : weights_)
    {
        if (m)
        {
            l2 += m->dotSelf();
        }
    }
    return l2;
}

void Net::calNorm(int& n, realc& l1, realc& l2) const
{
    n = weightDataSize();
    l1 = weightSumAbs();
    l2 = weightNorm2();
}

void Net::outputNorm() const
{
    int n;
    realc l1, l2;
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
            m->resizeNumber(n);
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
            m->resizeNumber(n);
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
    std::vector<int> group;
    if (op_queue_.back().getType() == MatrixOpType::CONCAT)
    {
        for (auto& m : op_queue_.back().getMatrixIn())
        {
            group.push_back(m->getRow());
        }
    }
    return group;
}

//测试一个大组
//test_type时，result返回准确率，否则返回的是error
//返回值为0表示基本正常，为负表示输入不合法，为正表示网络的测试结果只有全对或全错，此时一般存在数值问题
int Net::test(const std::string& info, Matrix* X, Matrix* Y, Matrix* A, int output_group /*= 0*/, int test_type /*= 0*/, int attack_times /*= 0*/, realc* result /*= nullptr*/)
{
    if (info != "")
    {
        LOG("{}, {} groups of data...\n", info, X->getNumber());
    }
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

    int group_size = X->getNumber();
    if (group_size <= 0)
    {
        return 0;
    }

    bool need_y = output_group > 0 || test_type > 0 || result;

    setActivePhase(ACTIVE_PHASE_TEST);

    realc total_error = 0, error = 0;
    realc* errorp = nullptr;
    if (test_type == 0 && result)
    {
        errorp = &error;
    }

    //此处注意二者只会有一个为空
    //在执行时，Y可以为空，但是A不能为空
    Matrix temp;
    if (A == nullptr)
    {
        temp.resize(*Y);
        A = &temp;
    }

    auto Xp = getX(), Yp = getY(), Ap = getA();
    int batch = getX().getNumber();
    for (int i = 0; i < group_size; i += batch)
    {
        //检查最后一组是不是组数不足
        int n_rest = std::min(batch, group_size - i);
        if (n_rest < batch)
        {
            //getX().resizeNumber(n_rest);
            //getY().resizeNumber(n_rest);
            //getA().resizeNumber(n_rest);
            resetBatchSize(n_rest);
        }
        //todo: 需适配amd
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

        active(nullptr, nullptr, nullptr, false, errorp);

        if (attack_times)
        {
            if (!X->inGpu())
            {
                Matrix::copyRows(getX(), 0, *X, i, n_rest);
            }
        }
        if (A)
        {
            if (!A->inGpu())
            {
                Matrix::copyRows(getA(), 0, *A, i, n_rest);
            }
        }
        for (int gen_time = 0; gen_time < attack_times; gen_time++)
        {
            //attack(&X_sub, &Y_sub);
        }
        if (errorp)
        {
            total_error += error * getX().getNumber();
        }
    }
    //还原，如不还原则传入矩阵的显存可能不被释放
    resetBatchSize(batch);
    getX() = Xp;
    getY() = Yp;
    getA() = Ap;
    if (test_type == 0 && result)
    {
        total_error /= X->getNumber();
        *result = total_error;
        ConsoleControl::setColor(CONSOLE_COLOR_LIGHT_RED);
        LOG("Real error = {}\n", total_error);
        ConsoleControl::setColor(CONSOLE_COLOR_NONE);
    }

    //恢复网络原来的设置
    setActivePhase(ACTIVE_PHASE_TRAIN);

    //若Y为空没有继续测试的必要
    if (Y == nullptr)
    {
        return 0;
    }

    int ret = 0;
    if (test_type == 1)
    {
        int y_size = Y->getRow();
        auto Y_cpu = Y->autoShareClone(UnitType::CPU);
        auto A_cpu = A->autoShareClone(UnitType::CPU);

        if (output_group > 0)
        {
            LOG("Label --> Infer\n");
            for (int i = 0; i < (std::min)(group_size, output_group); i++)
            {
                for (int j = 0; j < y_size; j++)
                {
                    LOG("{:6.3f} ", Y_cpu.getData(j, i));
                }
                LOG(" --> ");
                for (int j = 0; j < y_size; j++)
                {
                    LOG("{:6.3f} ", A_cpu.getData(j, i));
                }
                LOG("\n");
            }
        }

        ConsoleControl::setColor(CONSOLE_COLOR_LIGHT_RED);
        auto A_max = Matrix(A_cpu.getDim(), UnitType::CPU);

        //查看最后一层是否是拼起来的
        auto group = getTestGroup();
        if (group.size() > 0)
        {
            A_max.fillData(0);
            for (int i_group = 0; i_group < A_cpu.getNumber(); i_group++)
            {
                int total_loc = 0;
                for (int i_combine = 0; i_combine < group.size(); i_combine++)
                {
                    int out = group[i_combine];
                    if (out > 1)
                    {
                        real max_v = -9999;
                        int max_loc = 0;
                        for (int i = 0; i < out; i++)
                        {
                            real v = A_cpu.getData(total_loc + i, i_group);
                            if (v > max_v)
                            {
                                max_v = v;
                                max_loc = i;
                            }
                        }
                        A_max.getData(total_loc + max_loc, i_group) = 1;
                    }
                    else if (out == 1)
                    {
                        A_max.getData(total_loc, i_group) = A_cpu.getData(total_loc, i_group) > 0.5 ? 1 : 0;
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

        for (int i = 0; i < (std::min)(group_size, output_group); i++)
        {
            int o = A_max.indexColMaxAbs(i);
            int e = Y->indexColMaxAbs(i);
            LOG("{:3} ({:6.3f}) --> {:3}\n", o, A_cpu.getData(o, i), e);
        }
        std::vector<int> right(y_size, 0), total(y_size, 0);
        int right_total = 0;
        int total_total = 0;
        for (int i = 0; i < group_size; i++)
        {
            for (int j = 0; j < y_size; j++)
            {
                if (Y_cpu.getData(j, i) == 1)
                {
                    total[j]++;
                    total_total++;
                    if (A_max.getData(j, i) == 1)
                    {
                        right[j]++;
                        right_total++;
                    }
                }
            }
        }

        double accuracy_total = 1.0 * right_total / total_total;
        LOG("Total accuracy: {:.2f}% ({}/{}) (error/total)\n", 100 * accuracy_total, total_total - right_total, total_total);

        int i_group = 0;
        int total_group = 0;
        int count0 = 0, count1 = 0;
        for (int i = 0; i < y_size; i++)
        {
            if (right[i] == total[i])    //若total为0也视为正确
            {
                count1++;
            }
            else if (right[i] == 0)
            {
                count0++;
            }
            double accur = 100.0 * right[i] / total[i];
            LOG("{}: {:.2f}% ({}/{})", i, accur, total[i] - right[i], total[i]);
            if (group.size() > i_group && i == total_group + group[i_group] - 1)
            {
                LOG("\n");
                total_group += group[i_group];
                i_group++;
            }
            else if (i < y_size - 1)
            {
                LOG(", ");
            }
            else
            {
                LOG("\n");
            }
        }
        //若准确率只有1或0，则表示该结果有问题
        //但若全是1，则没有问题
        if (count0 > 0 && count0 + count1 == y_size)
        {
            ret = 1;
        }
        ConsoleControl::setColor(CONSOLE_COLOR_NONE);
        if (result)
        {
            *result = accuracy_total;
        }
    }
    else if (test_type == 2)
    {
        auto Y_cpu = Y->autoShareClone(UnitType::CPU);
        auto A_cpu = A->autoShareClone(UnitType::CPU);

        LOG("Label --> Infer\n");
        //std::string chars = R"( `.^,:~"<!ct+{i7?u30pw4A8DX%#HWM)";
        std::string chars = " .oO";
        auto out_char = [&chars](float& f)
        {
            const int m = chars.size();
            int n = f * m;
            n = std::max(0, std::min(m - 1, n));
            return chars[n];
        };
        auto out_matrix = [&out_char](Matrix& M1, Matrix& M2, int n)
        {
            int step = std::max(1, M1.getWidth() / 32);
            for (int iw = 0; iw < M1.getWidth(); iw += step)
            {
                LOG("[");
                for (int ih = 0; ih < M1.getHeight(); ih += step)
                {
                    LOG("{}", out_char(M1.getData(iw, ih, 0, n)));
                }
                LOG("]         [");
                for (int ih = 0; ih < M2.getHeight(); ih += step)
                {
                    LOG("{}", out_char(M2.getData(iw, ih, 0, n)));
                }
                LOG("]\n");
            }
        };
        out_matrix(Y_cpu, A_cpu, 0);

        std::vector<int64_t> right(2, 0), total(2, 0);
        for (int64_t i = 0; i < Y_cpu.getDataSize(); i++)
        {
            if (Y_cpu.getData(i) < 0.5)
            {
                total[0]++;
                if (A_cpu.getData(i) < 0.5)
                {
                    right[0]++;
                }
            }
            else if (Y_cpu.getData(i) >= 0.5)
            {
                total[1]++;
                if (A_cpu.getData(i) >= 0.5)
                {
                    right[1]++;
                }
            }
        }
        double accuracy_total = 1.0 * (right[0] + right[1]) / Y_cpu.getDataSize();
        LOG("Total accuracy: {:.2f}% ({}/{}) (error/total)\n", 100 * accuracy_total, Y_cpu.getDataSize() - right[0] - right[1], Y_cpu.getDataSize());
        for (int i = 0; i < 2; i++)
        {
            double accur = 100.0 * right[i] / total[i];
            LOG("{}: {:.2f}% ({}/{})", i, accur, total[i] - right[i], total[i]);
            if (i < 2 - 1)
            {
                LOG(", ");
            }
            else
            {
                LOG("\n");
            }
        }
    }
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
        if (m)
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
    LOG("Initialized weight method is {}\n", option_->getStringFromEnum(filler));

    for (auto& op : op_queue_)
    {
        for (auto& m : op.getMatrixWb())
        {
            if (!VectorMath::vector_have(weights_, m.get()))
            {
                int one_channel = m->getRow() / m->getChannel();
                MatrixEx::fill(*m, filler, m->getChannel() * one_channel, m->getNumber() * one_channel);
                //m->scale(10);    //调试用
                weights_.push_back(m.get());
            }
        }
    }
    combineWeights(weights_, all_weights_);
}

std::string Net::ir()
{
    std::string ir;
    for (auto& w : weights_)
    {
        ir += fmt1::format("M{} = {};", w, w->sizeMessage());
    }
    ir += MatrixOp::ir(op_queue_);
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

//获取一个大组的攻击样本
void Net::groupAttack(Matrix* X, Matrix* Y, int attack_times)
{
    test("Attack net", X, Y, nullptr, 0, 0, attack_times, nullptr);
}

void Net::adjustByEpoch(int epoch, int total_epoch)
{
    solver_.adjustLearnRate(epoch, total_epoch);
}

//Net* Net::clone(int clone_data /*= 0*/)
//{
//    auto net = new Net();
//    net->setOption(option_);
//    net->setBatch(batch_);
//    net->init();
//    if (clone_data)
//    {
//        SaveBuffer buffer;
//        save(buffer);
//        buffer.resetPointer();
//        net->load(buffer);
//    }
//    return net;
//}

}    // namespace cccc