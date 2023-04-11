#include "Net.h"
#include "ConsoleControl.h"
#include "filefunc.h"
#include "Layer.h"
#include "Timer.h"
#include "VectorMath.h"
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

    getX().setNeedReverse(false);
    initWeights();
    solver_.init(option_, "", all_weights_);

    if (loss_.empty())    //未设置损失函数就自己添加一个
    {
        MatrixOp op;
        if (loss_weight_.getDataSize() == 0)
        {
            op.set(MatrixOpType::LOSS, { A_, Y_ }, {}, {});
        }
        else
        {
            loss_weight_.resizeNumber(A_->getNumber());
            loss_weight_.repeat(1);
            op.set(MatrixOpType::LOSS, { A_, Y_ }, {}, {}, {}, {}, { loss_weight_ });
        }
        loss_.push_back(op);
        //LossWeight_->printAsMatrix();
    }

    batches_for_learn_ = option_->getInt("train", "batches_for_learn", 1);

    //计算总占用空间
    //std::map<void*, int64_t> map1;
    //int64_t max_size = 0, sum = 0;
    //for (auto& op : op_queue_)
    //{
    //    for (auto& m : op.getMatrixIn())
    //    {
    //        map1[m->getDataPointer()] = std::max(m->getDataSizeInByte(), map1[m->getDataPointer()]);
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

//learn为真时，会反向更新网络
//active只处理一个gpu中的minibatch
//A是外部提供的矩阵，用于保存结果
void Net::active(Matrix* X, Matrix* Y, Matrix* A, bool learn, realc* error)
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
    //X->message();
    //getX().message();
    //getY().message();
    //op_queue_[0].getMatrixIn()[0]->message();
    MatrixOp::forward(op_queue_);
    if (learn)
    {
        //all_weights_.d().message("d0");
        MatrixOp::backward(op_queue_, loss_, true);
        //all_weights_.d().message("d1");
        //all_weights_.message("0");
        //LOG("%d\n", getX().getNumber());
        if ((learned_batches_ + 1) % batches_for_learn_ == 0)
        {
            solver_.updateWeights(getX().getNumber() * batches_for_learn_);
            //all_weights_.d().message("1");
            solver_.actMomentum();
            //all_weights_.d().message("momentum");
        }
        learned_batches_++;
    }
    if (A)
    {
        Matrix::copyDataPointer(getA(), getA().getDataPointer(), *A, A->getDataPointer(), getA().getDataSize());
    }
    if (error && Y)
    {
        Matrix R(getA().getDim());
        Matrix::add(getA(), *Y, R, 1, -1);
        *error = R.dotSelf() / Y->getNumber();
    }
}

//保存权重，需配合ini中的网络结构
//返回值：0正常，其他值不正常
int Net::saveWeight(const std::string& filename)
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

    std::string suffix;
    if (!option_->getString("train", "save_sign").empty())
    {
        suffix = option_->dealString(option_->getString("train", "save_sign")) + " " + Timer::getNowAsString();
        buffer += "save_sign\n" + suffix + "\n";
    }

    if (strfunc::writeStringToFile(buffer, filename) > 0)
    {
        LOG("done\n");
        LOG("Save sign: {}\n", suffix);
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
        LOG("Warning: no data!\n");
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

    auto p = (real*)buffer.data();
    int sum = 0;
    for (auto& m : weights_)
    {
        sum += m->load(p + sum, (buffer.size() - sum * sizeof(real)) / sizeof(real));
    }

    LOG("done\n");

    int ret = 0;
    std::string sign_substr = "save_sign\n";
    auto weght_end_pos = buffer.find(sign_substr, buffer.size() - 100);    //存档签名不能超过100个字节
    if (weght_end_pos != std::string::npos)
    {
        auto sign_begin_pos = weght_end_pos + sign_substr.size();
        std::string sign;
        auto sign_end_pos = buffer.find("\n", sign_begin_pos);
        if (sign_end_pos != std::string::npos)
        {
            sign = buffer.substr(sign_begin_pos, sign_end_pos - sign_begin_pos);
        }
        LOG("Save sign: {}\n", sign);
    }
    else
    {
        LOG("Warning: no save sign!\n");
        weght_end_pos = buffer.size();
        ret = 1;
    }
    if (weght_end_pos > sum * sizeof(real))
    {
        LOG("Warning: size of weight is longer than net!\n");
        ret = 1;
    }
    else if (weght_end_pos < sum * sizeof(real))
    {
        LOG("Warning: size of weight is shorter than net!\n");
        ret = 1;
    }
    return ret;
}

//计算网络中参数的的L1和L2范数
void Net::calNorm(realc& l1, realc& l2)
{
    setDeviceSelf();
    l1 = 0, l2 = 0;
    //计算L1和L2
    for (auto& m : weights_)
    {
        if (m)
        {
            l1 += m->sumAbs();
            l2 += m->dotSelf();
        }
    }
}

//返回值非零表示网络已经出现数值问题
int Net::checkNorm()
{
    realc l1, l2;
    calNorm(l1, l2);
    LOG("L1 = {}, L2 = {}\n", l1, l2);
    return isnan(l1) || isnan(l2);
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
//返回值是max位置准确率，即标签正确率
//test_type时，result返回准确率，否则返回的是error
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
    if (test_type == 2)
    {
        group_size = 1;
    }
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
        if (X->inGPU())
        {
            getX().shareData(*X, 0, i);
        }
        else
        {
            Matrix::copyRows(*X, i, getX(), 0, n_rest);
        }
        if (Y)
        {
            if (Y->inGPU())
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
            if (A->inGPU())
            {
                getA().shareData(*A, 0, i);
            }
        }

        active(nullptr, nullptr, nullptr, false, errorp);

        if (attack_times)
        {
            if (!X->inGPU())
            {
                Matrix::copyRows(getX(), 0, *X, i, n_rest);
            }
        }
        if (A)
        {
            if (!A->inGPU())
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

    if (test_type == 1)
    {
        int y_size = Y->getRow();
        auto Y_cpu = Y->autoShareClone(DeviceType::CPU);
        auto A_cpu = A->autoShareClone(DeviceType::CPU);

        LOG("Label -> Infer\n");
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

        ConsoleControl::setColor(CONSOLE_COLOR_LIGHT_RED);
        auto A_max = Matrix(A_cpu.getDim(), DeviceType::CPU);

        //查看最后一层是否是拼起来的
        auto group = getTestGroup();
        if (group.size() > 0)
        {
            A_max.initData(0);
            for (int i_group = 0; i_group < A_cpu.getNumber(); i_group++)
            {
                int total_loc = 0;
                for (int i_combine = 0; i_combine < group.size(); i_combine++)
                {
                    real max_v = -9999;
                    int max_loc = 0;
                    int out = group[i_combine];
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
        std::vector<int> right(y_size), total(y_size);
        for (int j = 0; j < y_size; j++)
        {
            right[j] = 0;
            total[j] = 0;
        }
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
        for (int j = 0; j < y_size; j++)
        {
            double accur = 100.0 * right[j] / total[j];
            LOG("{}: {:.2f}% ({}/{})", j, accur, total[j] - right[j], total[j]);
            if (group.size() > i_group && j == total_group + group[i_group] - 1)
            {
                LOG("\n");
                total_group += group[i_group];
                i_group++;
            }
            else if (j < y_size - 1)
            {
                LOG(", ");
            }
            else
            {
                LOG("\n");
            }
        }
        ConsoleControl::setColor(CONSOLE_COLOR_NONE);
        if (result)
        {
            *result = accuracy_total;
        }
    }
    else if (test_type == 2)
    {
        Matrix Y_cpu = Y->cloneSharedCol();
        Matrix A_cpu = A->cloneSharedCol();
        Y_cpu.toCPU();
        A_cpu.toCPU();

        LOG("Label -> Infer\n");
        auto out_matrix = [](Matrix& M1, Matrix& M2)
        {
            int step = 4;
            for (int iw = 0; iw < M1.getWidth(); iw += step)
            {
                LOG("[");
                for (int ih = 0; ih < M1.getHeight(); ih += step)
                {
                    auto v = M1.getData(iw, ih, 0, 0);
                    if (v > 0.5)
                    {
                        LOG("#");
                    }
                    else
                    {
                        LOG(" ");
                    }
                }
                LOG("]         [");
                for (int ih = 0; ih < M2.getHeight(); ih += step)
                {
                    auto v = M2.getData(iw, ih, 0, 0);
                    if (v > 0.5)
                    {
                        LOG("#");
                    }
                    else
                    {
                        LOG(" ");
                    }
                }
                LOG("]\n");
            }
        };

        out_matrix(Y_cpu, A_cpu);
    }
    return 0;
}

//将所有参数集中在同一块内存，方便并行中的数据交换
void Net::combineWeights()
{
    setDeviceSelf();

    //此处注意在单卡情况下，合并变量有可能会变慢，原因可能是对齐的内存会比较有效率
    auto c256 = [](int64_t i)
    { return (i + 255) / 256 * 256; };

    int64_t sum = 0;
    for (int i = 0; i < weights_.size(); i++)
    {
        auto m = weights_[i];
        sum += c256(m->getDataSize());
    }
    all_weights_.resize(1, sum);
    auto& dparameters = all_weights_.d();
    all_weights_.initData(0);
    dparameters.initData(0);
    int64_t p = 0;
    int mode = 2;
    for (int i = 0; i < weights_.size(); i++)
    {
        auto& m = weights_[i];
        if (m)
        {
            if (mode == 0 || mode == 2)
            {
                auto m1 = m->clone();
                m->shareData(all_weights_, 0, p);
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
    workspace_.resize(all_weights_);
}

void Net::initWeights()
{
    auto filler = option_->getEnum("", "init_weight", RANDOM_FILL_XAVIER);
    LOG("Initialized weight method is {}\n", option_->getStringFromEnum(filler));
    for (auto& op : op_queue_)
    {
        for (auto& m : op.getMatrixWb())
        {
            int one_channel = m->getRow() / m->getChannel();
            MatrixEx::fill(*m, filler, m->getChannel() * one_channel, m->getNumber() * one_channel);
            //m->scale(10);    //调试用
            weights_.push_back(m.get());
        }
    }
    combineWeights();
}

//利用已知网络修改X适应答案，只处理一个minibatch
void Net::attack(Matrix* X, Matrix* Y)
{
    active(X, Y, nullptr, false, nullptr);

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