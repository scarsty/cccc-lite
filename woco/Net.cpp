#include "Net.h"
#include "ConsoleControl.h"
#include "File.h"
#include "Log.h"
#include "MatrixOperator.h"
#include "Timer.h"
#include "VectorMath.h"
#include <algorithm>

namespace woco
{

Net::Net()
{
    device_id_ = CudaControl::getCurrentDevice();
}

void Net::makeStructure()
{
    setDeviceSelf();
    MatrixOperator::beginMaking();
    structure();
    MatrixOperator::endMaking();
    std::swap(MatrixOperator::getQueue(), op_queue_);    //移动避免复制操作
    //op_queue_back_ = op_queue_ + loss_;
    //std::reverse(op_queue_back_.begin(), op_queue_back_.end());
    if (solvers_.size() == 0)
    {
        solvers_.resize(weights_.size());
        for (int i = 0; i < solvers_.size(); i++)
        {
            solvers_[i].setWeight(weights_[i]);
        }
    }
    //MatrixOperator::simpleQueue(op_queue_, X_, A_);
    MatrixOperator::print(op_queue_);
    MatrixOperator::print(loss_);
    X_.setNeedReverse(false);    //训练权重时X不需反向，可能是不对待查
}

void Net::structure()
{
    A_ = X_;
    Y_ = A_;
    loss_.clear();
}

void Net::forward()
{
    MatrixOperator::forward(op_queue_);
}

void Net::backward()
{
    MatrixOperator::backward(op_queue_, loss_, workspace_back_);
    for (int i = 0; i < solvers_.size(); i++)
    {
        solvers_[i].updateWeight(X_.getNumber());
    }
}

void Net::setXYA(const Matrix& X, const Matrix& Y, const Matrix& A)
{
    X_ = X;
    Y_ = Y;
    A_ = A;
}

//learn为真时，会反向更新网络
//active只处理一个gpu中的minibatch，需预先设置好网络的X_，Y_，A_
void Net::active(bool learn)
{
    //setDeviceSelf();
    setActivePhase(learn ? ACTIVE_PHASE_TRAIN : ACTIVE_PHASE_TEST);
    forward();
    if (learn)
    {
        backward();
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
    Log::LOG("Save net to %s... ", filename.c_str());

    int sum = 0;
    for (auto& m : weights_)
    {
        sum += m.getDataSize();
    }
    std::string buffer(sum, '\0');
    auto p = (real*)buffer.data();
    for (auto& m : weights_)
    {
        p += m.save(p, m.getDataSize());
    }

    //if (!Option::getString("", "save_sign").empty())
    //{
    //    std::string suffix = "save_sign\n" + option_->getString("", "save_sign") + " " + Timer::getNowAsString() + "\n";
    //    str += suffix;
    //}

    if (convert::writeStringToFile(buffer, filename) > 0)
    {
        Log::LOG("done\n");
        return 0;
    }
    else
    {
        Log::LOG("failed!\n");
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
        Log::LOG("Warning: no data!\n");
        return -2;
    }

    if (load_mode == 0)
    {
        Log::LOG("Loading net from %s... ", str.c_str());
    }
    else
    {
        Log::LOG("Loading net from memory... ");
    }

    std::string buffer;
    if (File::fileExist(str))
    {
        buffer = convert::readStringFromFile(str);
    }
    else if (load_mode)
    {
        buffer = str;
    }

    if (buffer.size() <= 0)
    {
        Log::LOG("failed!\n");
        return -2;
    }

    auto p = (real*)buffer.data();
    int sum = 0;
    for (auto& m : weights_)
    {
        sum += m.load(p + sum, (buffer.size() - sum) / sizeof(real));
    }

    Log::LOG("done\n");

    int ret = 0;
    std::string sign_substr = "save_sign\n";
    auto weght_end_pos = str.find(sign_substr, str.size() - 100);    //存档签名不能超过100个字节
    if (weght_end_pos != std::string::npos)
    {
        auto sign_begin_pos = weght_end_pos + sign_substr.size();
        std::string sign;
        auto sign_end_pos = str.find("\n", sign_begin_pos);
        if (sign_end_pos != std::string::npos)
        {
            sign = str.substr(sign_begin_pos, sign_end_pos - sign_begin_pos);
        }
        Log::LOG("Save sign: %s\n", sign.c_str());
    }
    else
    {
        Log::LOG("Warning: no save sign!\n");
        weght_end_pos = str.size();
        ret = 1;
    }
    if (weght_end_pos > str.size())
    {
        Log::LOG("Warning: size of weight is longer than net!\n");
        ret = 1;
    }
    else if (weght_end_pos < str.size())
    {
        Log::LOG("Warning: size of weight is shorter than net!\n");
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
    for (auto m : weights_)
    {
        l1 += m.sumAbs();
        l2 += m.dotSelf();
    }
    Log::LOG("L1 = %g, L2 = %g\n", l1, l2);
}

//设置数据组数
int Net::resetBatchSize(int n)
{
    //对于一个网络，所有层数据组数应该一致
    //if (n == layer_vector_[0]->getBatchSize())
    //{
    //    return n;
    //}

    //for (auto l : layer_vector_)
    //{
    //    l->resetGroupCount(n);
    //}

    return n;
}

//调整整个网络的学习率
realc Net::adjustLearnRate(int ec)
{
    realc lr = 0;
    //for (auto l : layer_vector_)
    //{
    //    lr = l->adjustLearnRate(ec);
    //}
    return lr;
}

//测试一个大组
//返回值是max位置准确率，即标签正确率
//test_max时，result返回准确率，否则返回的是error
int Net::test(const std::string& info, Matrix& X, Matrix& Y, Matrix& A, int output_group, int test_max, int attack_times)
{
    if (info != "")
    {
        Log::LOG("%s, %d groups of data...\n", info.c_str(), X.getNumber());
    }

    setDeviceSelf();

    if (X.getNumber() != Y.getNumber())
    {
        return -1;
    }

    int group_size = X.getNumber();
    if (group_size <= 0)
    {
        return 0;
    }

    A.resize(Y);
    bool need_y = output_group > 0 || test_max > 0;
    setActivePhase(ACTIVE_PHASE_TEST);
    Matrix X_gpu(DeviceType::GPU), Y_gpu(DeviceType::GPU), A_gpu(DeviceType::GPU);
    if (X.getDeviceType() == DeviceType::GPU)
    {
        X_gpu = X;
    }
    else
    {
        X_gpu = X.clone(DeviceType::GPU);
    }
    if (Y.getDeviceType() == DeviceType::GPU)
    {
        Y_gpu = Y;
    }
    else
    {
        Y_gpu = Y.clone(DeviceType::GPU);
    }
    if (A.getDeviceType() == DeviceType::GPU)
    {
        A_gpu = A;
    }
    else
    {
        A_gpu = A.clone(DeviceType::GPU);
    }

    realc total_error = 0, error = 0;
    realc* errorp = nullptr;
    if (test_max == 0)
    {
        errorp = &error;
    }
    for (int i = 0; i < group_size; i += X_.getNumber())
    {
        //检查最后一组是不是组数不足,unfinished
        int n_rest = group_size - i;
        //if (n_rest < batch_)
        //{
        //    data_sub.resizeNumber(n_rest);
        //    resetBatchSize(n_rest);
        //}
        X_.shareData(X_gpu, 0, i);
        Y_.shareData(Y_gpu, 0, i);
        A_.shareData(A_gpu, 0, i);
        active(false);
        for (int gen_time = 0; gen_time < attack_times; gen_time++)
        {
            attack(X_, Y_);
        }
        if (errorp)
        {
            total_error += error * X_.getNumber();
        }
    }

    if (test_max == 0)
    {
        total_error /= X_.getNumber();
        //*result = total_error;
        ConsoleControl::setColor(CONSOLE_COLOR_LIGHT_RED);
        Log::LOG("Real error = %e\n", total_error);
        ConsoleControl::setColor(CONSOLE_COLOR_NONE);
    }
    if (attack_times)
    {
        Matrix::copyData(X_gpu, X);
    }
    Matrix::copyData(A_gpu, A);

    //恢复网络原来的设置组数
    //resetBatchSize(batch_);    //暂时不处理

    //setActivePhase(ACTIVE_PHASE_TRAIN);////////////////////////

    if (output_group == 0 && test_max == 0)
    {
        return 0;
    }

    int y_size = Y.getRow();
    auto Y_cpu = Y.clone(DeviceType::CPU);
    auto A_cpu = A_gpu.clone(DeviceType::CPU);

    for (int i = 0; i < (std::min)(group_size, output_group); i++)
    {
        for (int j = 0; j < y_size; j++)
        {
            Log::LOG("%6.3f ", A_cpu.getData(j, i));
        }
        Log::LOG(" --> ");
        for (int j = 0; j < y_size; j++)
        {
            Log::LOG("%6.3f ", Y_cpu.getData(j, i));
        }
        Log::LOG("\n");
    }

    if (test_max)
    {
        ConsoleControl::setColor(CONSOLE_COLOR_LIGHT_RED);
        Matrix A_max(A_cpu.getDim(), DeviceType::CPU);

        //查看最后一层是否是拼起来的
        //auto layer_out = getLastLayer();
        if (op_queue_.back().getType() == MatrixOpType::CONCAT)
        {
            //    A_max.initData(0);
            //    for (int i_group = 0; i_group < A_cpu.getCol(); i_group++)
            //    {
            //        int total_loc = 0;
            //        for (int i_combine = 0; i_combine < layer_out->getPrevLayers().size(); i_combine++)
            //        {
            //            real max_v = -9999;
            //            int max_loc = 0;
            //            int out = layer_out->getPrevLayers()[i_combine]->getOutTotal();
            //            for (int i = 0; i < out; i++)
            //            {
            //                real v = A_cpu.getData(total_loc + i, i_group);
            //                if (v > max_v)
            //                {
            //                    max_v = v;
            //                    max_loc = i;
            //                }
            //            }
            //            A_max->getData(total_loc + max_loc, i_group) = 1;
            //            total_loc += out;
            //        }
            //    }
        }
        else
        {
            A_max.initData(0);
            for (int i_group = 0; i_group < A_cpu.getCol(); i_group++)
            {
                real max_v = -9999;
                int max_loc = 0;
                for (int i = 0; i < A_cpu.getRow(); i++)
                {
                    real v = A_cpu.getData(i, i_group);
                    if (v > max_v)
                    {
                        max_v = v;
                        max_loc = i;
                    }
                }
                A_max.getData(max_loc, i_group) = 1;
            }
        }

        for (int i = 0; i < (std::min)(group_size, output_group); i++)
        {
            int o = A_max.indexColMaxAbs(i);
            int e = Y.indexColMaxAbs(i);
            Log::LOG("%3d (%6.4f) --> %3d\n", o, A_cpu.getData(o, i), e);
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
        Log::LOG("Total accuracy: %.2f%% (%d/%d) (error/total)\n", 100 * accuracy_total, total_total - right_total, total_total);

        for (int j = 0; j < y_size; j++)
        {
            double accur = 100.0 * right[j] / total[j];
            Log::LOG("%d: %.2f%% (%d/%d)", j, accur, total[j] - right[j], total[j]);
            if (j != y_size - 1)
            {
                Log::LOG(", ");
            }
        }
        Log::LOG("\n");
        ConsoleControl::setColor(CONSOLE_COLOR_NONE);
    }
    return 0;
}

//将所有参数集中在同一块内存，方便并行中的数据交换
void Net::combineParameters()
{
    setDeviceSelf();
    int64_t total_size = 0;
    for (auto& m : weights_)
    {
        total_size += m.getDataSize();
    }
    combined_weight_.resize(1, total_size);
    workspace_weight_.resize(1, total_size);

    total_size = 0;
    for (auto& m : weights_)
    {
        Matrix::copyDataPointer(m, m.getDataPointer(), combined_weight_, combined_weight_.getDataPointer(0, total_size), m.getDataSize());
        m.shareData(combined_weight_, 0, total_size);
        total_size += m.getDataSize();
    }
}

//利用已知网络修改X适应答案，即所谓自攻击，只处理一个minibatch
void Net::attack(Matrix& X, Matrix& Y)
{
    //active(X, Y, A_, false);

    //for (int i_layer = getLayersCount() - 1; i_layer >= 0; i_layer--)
    //{
    //    layer_vector_[i_layer]->activeBackward();
    //}
    //getFirstLayer()->updateABackward();  unfinished
}

//获取一个大组的攻击样本
void Net::groupAttack(Matrix& X, Matrix& Y, int attack_times)
{
    Matrix A;
    test("Attack net", X, Y, A, 0, attack_times);
}

}    // namespace woco