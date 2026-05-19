#include "MatrixOp.h"
#include "MatrixEx.h"
#include "Timer.h"
#include "strfunc.h"
#include <fstream>
#include <functional>
#include <vector>

namespace cccc
{

void MatrixOp::forward(std::vector<MatrixOp>& op_queue)
{
    //Timer t;
    for (auto& op : op_queue)
    {
        if (op.connect_x_)
        {
            Timer t;
            op.forwardData();
            op.forward_time_ += t.getElapsedTime();
            op.forward_count_++;
        }
    }
    //LOG("forward time: {} s\n", t.getElapsedTime());
}

void MatrixOp::backward(std::vector<MatrixOp>& op_queue, std::vector<MatrixOp>& loss, bool clear_d)
{
    if (clear_d)
    {
        //Timer t;
        for (auto& op : op_queue)
        {
            for (auto& m : op.in_)
            {
                m->setKeepWeight(0);
            }
        }
        for (auto& op : loss)
        {
            for (auto& m : op.in_)
            {
                m->setKeepWeight(0);
            }
        }
        //LOG("clear d time: {} s\n", t.getElapsedTime());
    }
    for (auto& op : loss)
    {
        Timer t;
        op.backwardLoss();
        op.backward_time_ += t.getElapsedTime();
        op.backward_count_++;
        //LOG("{}, {} loss time: {} s\n", getOpName(op.type_), op.index_, t.getElapsedTime());
    }
    for (auto it = op_queue.rbegin(); it != op_queue.rend(); ++it)
    {
        if (it->connect_loss_)
        {
            Timer t;
            it->backwardDataWeight();
            it->backward_time_ += t.getElapsedTime();
            it->backward_count_++;
            //LOG("{}, {} backward time: {}\n", getOpName(it->type_), it->index_, t.getElapsedTime());
            //it->out_[0]->d().message("dY" + getOpName(it->type_));
            //for (auto& m : it->in_)
            //{
            //    if (m->needBack())
            //    {
            //        m->message("X");
            //        m->d().message("dX" +getOpName(it->type_));
            //    }
            //}
        }
    }
}

void MatrixOp::forwardData()
{
    auto& X = *in_[0];
    auto& Y = *out_[0];
    switch (type_)
    {
    case MatrixOpType::SCALE:
        Matrix::scale(X, Y, a_[0]);
        break;
    case MatrixOpType::ADD:
    {
        Matrix::add(X, *in_[1], Y, a_[0], a_[1], b_[0]);
        for (int i = 2; i < (int)in_.size(); i++)
        {
            Matrix::add(Y, *in_[i], Y, 1, a_[i], 1);
        }
        break;
    }
    case MatrixOpType::MUL:
        Matrix::mul(X, *in_[1], Y, a_[0]);
        break;
    case MatrixOpType::BATCHED_MUL:
    {
        auto ta = anys_[0].to<MatrixTransType>();
        auto tb = anys_[1].to<MatrixTransType>();
        Matrix::mulBatched(X, *in_[1], Y, ta, tb, a_[0]);
        break;
    }
    case MatrixOpType::ELE_MUL:
        //参数in_[1]可以为单通道
        Matrix::elementMul(X, *in_[1], Y, a_[0], b_[0]);
        break;
    case MatrixOpType::ADD_BIAS:
        MatrixEx::addBias(X, *in_[1], Y, a_[0], b_[0]);
        break;
    case MatrixOpType::CONCAT:
        MatrixEx::concatByChannel(in_, Y);
        break;
    case MatrixOpType::ACTIVE:
        MatrixEx::activeForward(X, Y, anys_[0].to<ActiveFunctionType>(), anys_[1].to<std::vector<int>>(), anys_[2].to<std::vector<float>>());
        break;
    case MatrixOpType::POOL:
        MatrixEx::poolingForward(X, Y, anys_[0].to<PoolingType>(), anys_[1].to<PoolingReverseType>(), window_, stride_, padding_, a_[0], b_[0]);
        break;
    case MatrixOpType::CONV:
        MatrixEx::convolutionForward(X, *in_[1], Y, stride_, padding_, a_[0], b_[0]);
        break;
    case MatrixOpType::CORR:
        MatrixEx::correlationForward(X, *in_[1], Y, stride_, padding_, a_[0], b_[0]);
        break;
    case MatrixOpType::RESHAPE:
        Matrix::copyData(X, Y);
        break;
    case MatrixOpType::MAX:
        MatrixEx::matrix_max(*in_[0], *in_[1], Y);
        break;
    case MatrixOpType::BATCH_NORM:
    {
        auto& scale = *in_[1];
        auto& bias = *in_[2];
        auto bn_type = anys_[0].to<BatchNormalizationType>();
        //a_[0] = exp_aver_factor (mutable), a_[1] = epsilon
        MatrixEx::batchNormalizationForward(X, Y, bn_type, a_[0], a_[1], scale, bias);
        break;
    }
    case MatrixOpType::LAYER_NORM:
    {
        //in_[1]=scale, in_[2]=bias; a_[0]=epsilon
        auto& scale = *in_[1];
        auto& bias = *in_[2];
        MatrixEx::layerNormalizationForward(X, Y, scale, bias, a_[0]);
        break;
    }
    case MatrixOpType::POOL_CHANNEL:
        MatrixEx::poolingChannelForward(X, Y, anys_[0].to<PoolingType>(), anys_[1].to<PoolingReverseType>(), a_[0], b_[0]);
        break;
    case MatrixOpType::PREPEND_TOKEN:
    {
        //X: (D, T, 1, B), cls: (D, 1, 1, 1) -> Y: (D, T+1, 1, B)
        auto& cls = *in_[1];
        int D = X.getWidth();
        int T = X.getHeight();
        int B = X.getNumber();
        for (int n = 0; n < B; n++)
        {
            //token 0: 复制 cls
            Matrix::copyDataPtr(cls, cls.getDataPtr(0, 0, 0, 0), Y, Y.getDataPtr(0, 0, 0, n), D);
            //token 1..T: 复制 X 全部 token
            Matrix::copyDataPtr(X, X.getDataPtr(0, 0, 0, n), Y, Y.getDataPtr(0, 1, 0, n), D * T);
        }
        break;
    }
    case MatrixOpType::FIRST_TOKEN:
    {
        //X: (D, T, 1, B) -> Y: (D, 1, 1, B), 取每个 batch 的第 0 个 token
        int D = X.getWidth();
        int B = X.getNumber();
        for (int n = 0; n < B; n++)
        {
            Matrix::copyDataPtr(X, X.getDataPtr(0, 0, 0, n), Y, Y.getDataPtr(0, 0, 0, n), D);
        }
        break;
    }
    case MatrixOpType::RMS_NORM:
    {
        auto& scale = *in_[1];
        MatrixEx::rmsNormForward(X, Y, scale, a_[0]);
        break;
    }
    case MatrixOpType::PERMUTE:
    {
        auto& perm = anys_[0].to<std::vector<int>>();
        MatrixEx::permute4dForward(X, Y, perm);
        break;
    }
    case MatrixOpType::ROPE:
    {
        auto& cos_tab = *in_[1];
        auto& sin_tab = *in_[2];
        int pos_offset = window_.size() > 0 ? window_[0] : 0;
        MatrixEx::ropeForward(X, Y, cos_tab, sin_tab, pos_offset);
        break;
    }
    case MatrixOpType::ROPE_INTERLEAVED:
    {
        auto& cos_tab = *in_[1];
        auto& sin_tab = *in_[2];
        int pos_offset = window_.size() > 0 ? window_[0] : 0;
        MatrixEx::ropeInterleavedForward(X, Y, cos_tab, sin_tab, pos_offset);
        break;
    }
    case MatrixOpType::PIXEL_SHUFFLE:
    {
        int r = window_[0];
        MatrixEx::pixelShuffleForward(X, Y, r);
        break;
    }
    case MatrixOpType::KV_CACHE:
    {
        //in_[0]=X_new (D, T_new, H, B), in_[1]=cache (D, T_max, H, B); Y 已共享 cache 数据
        auto& cache = *in_[1];
        int D = X.getWidth();
        int T_new = X.getHeight();
        int H_heads = X.getChannel();
        int B = X.getNumber();
        int T_max = window_.size() > 1 ? window_[1] : cache.getHeight();
        int pos = window_.size() > 0 ? window_[0] : 0;
        if (pos + T_new > T_max)
        {
            //超出 cache 容量, 裁剪 (丢弃超部分)
            T_new = T_max - pos;
        }
        if (T_new > 0)
        {
            for (int n = 0; n < B; n++)
            {
                for (int c = 0; c < H_heads; c++)
                {
                    //每 (n,c) 切片: X_new (D, T_new) 连续; cache (D, T_max) 中 h=pos..pos+T_new 连续
                    Matrix::copyDataPtr(X, X.getDataPtr(0, 0, c, n),
                        cache, cache.getDataPtr(0, pos, c, n), D * T_new);
                }
            }
            window_[0] = pos + T_new;
        }
        break;
    }
    case MatrixOpType::PRINT_RATIO:
    {
        // 每 600 次 forward 打印一次（对应 MNIST batch=100 的大约每个 epoch）
        if (forward_count_ % 600 == 0)
        {
            auto& A = *in_[0];    // attention 输出
            auto& B = *in_[1];    // CNN 原始特征
            std::string label = anys_.size() > 0 ? anys_[0].to<std::string>() : "ratio";
            double rmsA = std::sqrt((double)A.dotSelf() / std::max((int64_t)1, A.getDataSize()));
            double rmsB = std::sqrt((double)B.dotSelf() / std::max((int64_t)1, B.getDataSize()));
            double ratio = (rmsB > 1e-10) ? rmsA / rmsB : 0.0;
            LOG("  [{}] attn_rms={:.4f} cnn_rms={:.4f} ratio={:.4f}\n", label, rmsA, rmsB, ratio);
        }
        break;
    }
    case MatrixOpType::PRINT_MESSAGE:
    {
        auto& X = *in_[0];
        std::string label = anys_.size() > 0 ? anys_[0].to<std::string>() : "";
        X.message(label);
        break;
    }
    case MatrixOpType::ATTENTION:
    {
        float dk = anys_[0].to<float>();
        int causal = (anys_.size() > 1) ? anys_[1].to<int>() : 0;
        int pos_offset = window_.size() > 0 ? window_[0] : 0;
        MatrixEx::attentionForward(*in_[0], *in_[1], *in_[2], Y, dk, causal, pos_offset);
        break;
    }
    case MatrixOpType::EMBED:
    {
        // in_[0]=ids (T,1,1,B), in_[1]=W (D,1,1,V), out_[0]=Y (D,T,1,B)
        MatrixEx::embedForward(*in_[0], *in_[1], Y);
        break;
    }
    case MatrixOpType::TILE:
    {
        // in_[0]=X, out_[0]=Y; repeats 存于 window_
        MatrixEx::tileForward(*in_[0], Y, window_);
        break;
    }
    case MatrixOpType::DECONV:
        MatrixEx::deconvolutionForward(X, *in_[1], Y, stride_, padding_, a_[0], b_[0]);
        break;
    case MatrixOpType::GROUP_NORM:
    {
        auto& scale = *in_[1];
        auto& bias = *in_[2];
        int G = (int)anys_[0].to<int>();
        MatrixEx::groupNormForward(X, Y, scale, bias, G, a_[0]);
        break;
    }
    case MatrixOpType::REPARAM:
        MatrixEx::reparamForward(*in_[0], *in_[1], Y);
        break;
    case MatrixOpType::UPSAMPLE:
    {
        int sh = window_[0], sw = window_[1];
        bool bilinear = (int)anys_[0].to<int>() != 0;
        MatrixEx::upsampleForward(X, Y, sh, sw, bilinear);
        break;
    }
    case MatrixOpType::CHUNK:
    {
        // window_[0]=start_w, window_[1]=size_w
        int start_w = window_[0], size_w = window_[1];
        MatrixEx::chunkForward(X, Y, start_w, size_w);
        break;
    }
    case MatrixOpType::SIN_TIME_EMBED:
    {
        // in_[0]=t_scalar (1,1,1,B), out_[0]=Y (d,1,1,B)
        // a_[0]=base; window_[0]=d
        int d = window_[0];
        float base = a_[0];
        int B = in_[0]->getNumber();
        int half_d = d / 2;
        // 从 GPU/CPU 读取 t 标量 (每个 batch 一个值)
        std::vector<float> t_vals(B);
        if (in_[0]->getDeviceType() == UnitType::CPU)
        {
            for (int b = 0; b < B; b++)
            {
                t_vals[b] = in_[0]->getData(0, 0, 0, b);
            }
        }
        else
        {
            Matrix tmp_cpu({ 1, 1, 1, B }, in_[0]->getDataType(), UnitType::CPU);
            Matrix::copyDataAcrossDevice(*in_[0], tmp_cpu);
            for (int b = 0; b < B; b++)
            {
                t_vals[b] = tmp_cpu.getData(0, 0, 0, b);
            }
        }
        // 在 CPU 上计算嵌入, 再上传
        Matrix emb_cpu({ d, 1, 1, B }, in_[0]->getDataType(), UnitType::CPU);
        for (int b = 0; b < B; b++)
        {
            float t = t_vals[b];
            for (int i = 0; i < half_d; i++)
            {
                float freq = std::pow(base, -2.0f * i / d);
                emb_cpu.setData(i, 0, 0, b, std::cos(t * freq));
                emb_cpu.setData(i + half_d, 0, 0, b, std::sin(t * freq));
            }
        }
        Matrix::copyDataAcrossDevice(emb_cpu, Y);
        break;
    }
    case MatrixOpType::DEBUG_SAVE:
    {
        // 仅在首次 forward 保存（b_[0]==0 时保存并置为1）
        if (b_[0] > 0.0f)
        {
            break;
        }
        b_[0] = 1.0f;
        auto& X = *in_[0];
        std::string filename = anys_.size() > 0 ? anys_[0].to<std::string>() : "";
        if (filename.empty())
        {
            break;
        }
        int64_t n = X.getDataSize();
        Matrix cpu_x(X.getDataType(), UnitType::CPU);
        cpu_x.resize(X.getDim());
        Matrix::copyData(X, cpu_x);
        std::vector<float> buf((size_t)n);
        for (int64_t i = 0; i < n; i++)
        {
            buf[(size_t)i] = cpu_x.getData((int)i);
        }
        std::ofstream dbg_f(filename, std::ios::binary);
        if (dbg_f)
        {
            dbg_f.write(reinterpret_cast<const char*>(buf.data()), (std::streamsize)(n * sizeof(float)));
        }
        LOG("DEBUG_SAVE: saved {} elements to {}\n", n, filename);
        break;
    }
    }
}

void MatrixOp::backwardDataWeight()
{
    auto& Y = *out_[0];
    //float data_weight = 0;
    //若反向过程需更新多个矩阵，则在函数内部判断needUpdate
    //if (Y.isHip()) { data_weight = 0; }    //miopen只支持反向时beta为0
    switch (type_)
    {
    case MatrixOpType::SCALE:
        for (int i = 0; i < in_.size(); i++)
        {
            if (in_[i]->needBack())
            {
                Matrix::scale(Y.d(), in_[i]->d(), a_[i], in_[i]->keepWeight());
            }
        }
        break;
    case MatrixOpType::ADD:
        for (int i = 0; i < in_.size(); i++)
        {
            if (in_[i]->needBack())
            {
                Matrix::add(in_[i]->d(), Y.d(), in_[i]->d(), in_[i]->keepWeight(), a_[i]);
            }
        }
        break;
    case MatrixOpType::MUL:
        if (in_[1]->needBack())
        {
            Matrix::mul(*in_[0], Y.d(), in_[1]->d(), a_[0], in_[1]->keepWeight(), MATRIX_TRANS, MATRIX_NO_TRANS);
        }
        if (in_[0]->needBack())
        {
            Matrix::mul(Y.d(), *in_[1], in_[0]->d(), a_[0], in_[0]->keepWeight(), MATRIX_NO_TRANS, MATRIX_TRANS);
        }
        break;
    case MatrixOpType::BATCHED_MUL:
    {
        auto ta = anys_[0].to<MatrixTransType>();
        auto tb = anys_[1].to<MatrixTransType>();
        auto& A = *in_[0];
        auto& B = *in_[1];
        //dA: 形状与 A 物理布局一致
        if (in_[0]->needBack())
        {
            if (ta == MATRIX_NO_TRANS)
            {
                //dA(M,K) = dC(M,N) @ op(B)^T
                MatrixTransType tb_for = (tb == MATRIX_NO_TRANS) ? MATRIX_TRANS : MATRIX_NO_TRANS;
                Matrix::mulBatched(Y.d(), B, in_[0]->d(), MATRIX_NO_TRANS, tb_for, a_[0], in_[0]->keepWeight());
            }
            else
            {
                //ta=T, A 物理 (K,M), dA(K,M) = op(B) @ dC^T
                Matrix::mulBatched(B, Y.d(), in_[0]->d(), tb, MATRIX_TRANS, a_[0], in_[0]->keepWeight());
            }
        }
        //dB
        if (in_[1]->needBack())
        {
            if (tb == MATRIX_NO_TRANS)
            {
                //dB(K,N) = op(A)^T @ dC
                MatrixTransType ta_for = (ta == MATRIX_NO_TRANS) ? MATRIX_TRANS : MATRIX_NO_TRANS;
                Matrix::mulBatched(A, Y.d(), in_[1]->d(), ta_for, MATRIX_NO_TRANS, a_[0], in_[1]->keepWeight());
            }
            else
            {
                //tb=T, dB(N,K) = dC^T @ op(A)
                Matrix::mulBatched(Y.d(), A, in_[1]->d(), MATRIX_TRANS, ta, a_[0], in_[1]->keepWeight());
            }
        }
        break;
    }
    case MatrixOpType::ELE_MUL:
        if (in_[0]->needBack())
        {
            Matrix::elementMul(Y.d(), *in_[1], in_[0]->d(), a_[0], in_[0]->keepWeight());
        }
        if (in_[1]->needBack())
        {
            if (in_[1]->getChannel() == Y.getChannel())
            {
                Matrix::elementMul(Y.d(), *in_[0], in_[1]->d(), a_[0], in_[1]->keepWeight());
            }
            else if (in_[1]->getChannel() == 1)
            {
                MatrixEx::elementMulSum(Y.d(), *in_[0], in_[1]->d(), a_[0], in_[1]->keepWeight());
            }
        }
        break;
    case MatrixOpType::ADD_BIAS:
        if (in_[0]->needBack())
        {
            Matrix::add(in_[0]->d(), Y.d(), in_[0]->d(), 0, 1);
        }
        if (in_[1]->needBack())
        {
            MatrixEx::addBiasBackward(*in_[0], *in_[1], Y, 1, in_[1]->keepWeight());
        }
        break;
    case MatrixOpType::CONCAT:
        MatrixEx::concatByChannelBackward(in_, Y);
        break;
    case MatrixOpType::ACTIVE:
        if (in_[0]->needBack())
        {
            MatrixEx::activeBackward(*in_[0], Y, anys_[0].to<ActiveFunctionType>(), anys_[1].to<std::vector<int>>(), anys_[2].to<std::vector<float>>(), 1, in_[0]->keepWeight());
        }
        break;
    case MatrixOpType::POOL:
        if (in_[0]->needBack())
        {
            MatrixEx::poolingBackward(*in_[0], Y, anys_[0].to<PoolingType>(), anys_[1].to<PoolingReverseType>(), window_, stride_, padding_, a_[0], in_[0]->keepWeight());
        }
        break;
    case MatrixOpType::CONV:
        if (in_[0]->needBack())
        {
            MatrixEx::convolutionBackwardDX(*in_[0], *in_[1], Y, stride_, padding_, a_[0], in_[0]->keepWeight());
        }
        if (in_[1]->needBack())
        {
            MatrixEx::convolutionBackwardDW(*in_[0], *in_[1], Y, stride_, padding_, a_[0], in_[1]->keepWeight());
        }
        //MatrixEx::convolutionBackward(*in_[0], *wb_[0], Y, stride_, padding_, a_[0], in_[0]->keepWeight(), a[0], data_weight, &anys_[1].to<MatrixEx::ConvMethod>(), &anys_[2].to<MatrixEx::ConvMethod>(), &workspace_[1], &workspace_[2]);
        break;
    case MatrixOpType::CORR:
        //unfinished: 反向未实现，待补充correlationBackward调用
        //MatrixEx::correlationBackward(*in_[0], *wb_[0], Y, para_int_, para_matrix_, para_int_v_[0], para_int_v_[1], para_real_[0], in_[0]->keepWeight(), data_weight);
        break;
    case MatrixOpType::RESHAPE:
        Matrix::add(Y.d(), in_[0]->d(), in_[0]->d(), 1, in_[0]->keepWeight());
        break;
    case MatrixOpType::MAX:
        MatrixEx::matrix_maxb(*in_[0], *in_[1], Y, in_[0]->keepWeight(), in_[1]->keepWeight(), 1);
        break;
    case MatrixOpType::BATCH_NORM:
    {
        auto& scale = *in_[1];
        auto& bias = *in_[2];
        auto bn_type = anys_[0].to<BatchNormalizationType>();
        MatrixEx::batchNormalizationBackward(*in_[0], Y, bn_type, a_[1], scale, bias);
        //将梯度复制到对应的.d()矩阵，供Solver更新
        if (in_[1]->needBack())
        {
            Matrix::add(Y.getWorkspace(4), in_[1]->d(), in_[1]->d(), 1, in_[1]->keepWeight());
        }
        if (in_[2]->needBack())
        {
            Matrix::add(Y.getWorkspace(5), in_[2]->d(), in_[2]->d(), 1, in_[2]->keepWeight());
        }
        break;
    }
    case MatrixOpType::LAYER_NORM:
    {
        auto& scale = *in_[1];
        auto& bias = *in_[2];
        MatrixEx::layerNormalizationBackward(*in_[0], Y, scale, bias, a_[0]);
        if (in_[1]->needBack())
        {
            Matrix::add(Y.getWorkspace(2), in_[1]->d(), in_[1]->d(), 1, in_[1]->keepWeight());
        }
        if (in_[2]->needBack())
        {
            Matrix::add(Y.getWorkspace(3), in_[2]->d(), in_[2]->d(), 1, in_[2]->keepWeight());
        }
        break;
    }
    case MatrixOpType::POOL_CHANNEL:
        if (in_[0]->needBack())
        {
            MatrixEx::poolingChannelBackward(*in_[0], Y, anys_[0].to<PoolingType>(), anys_[1].to<PoolingReverseType>(), a_[0], in_[0]->keepWeight());
        }
        break;
    case MatrixOpType::PREPEND_TOKEN:
    {
        //Y: (D, T+1, 1, B), in_[0]=X (D,T,1,B), in_[1]=cls (D,1,1,1)
        auto& X = *in_[0];
        auto& cls = *in_[1];
        int D = X.getWidth();
        int T = X.getHeight();
        int B = X.getNumber();
        if (X.needBack())
        {
            //dX = dY[:, 1:T+1, :, :], 直接覆盖 (假设 keepWeight=0; ViT 中 X 不被复用)
            for (int n = 0; n < B; n++)
            {
                Matrix::copyDataPtr(Y.d(), Y.d().getDataPtr(0, 1, 0, n), X.d(), X.d().getDataPtr(0, 0, 0, n), D * T);
            }
        }
        if (cls.needBack())
        {
            //dcls = sum_n dY[:, 0, :, n]
            float kw = cls.keepWeight();
            if (kw == 0)
            {
                cls.d().fillData(0);
            }
            //单 sample (D,1,1,1) 视图; 用 cls.d() 的 shape/desc 即可
            Matrix dY_first = cls.d();
            for (int n = 0; n < B; n++)
            {
                dY_first.shareData(Y.d(), 0, 0, 0, n);
                Matrix::add(dY_first, cls.d(), cls.d(), 1, 1, 0);
            }
        }
        break;
    }
    case MatrixOpType::FIRST_TOKEN:
    {
        //Y: (D, 1, 1, B), in_[0]=X (D,T,1,B); dX[:, 0, :, n] = dY[:, 0, :, n], 其余 token 梯度为 0
        auto& X = *in_[0];
        if (X.needBack())
        {
            int D = X.getWidth();
            //先清零 dX (覆盖语义, 假设 keepWeight=0; ViT 中 X 不被复用)
            X.d().fillData(0);
            int B = X.getNumber();
            for (int n = 0; n < B; n++)
            {
                Matrix::copyDataPtr(Y.d(), Y.d().getDataPtr(0, 0, 0, n), X.d(), X.d().getDataPtr(0, 0, 0, n), D);
            }
        }
        break;
    }
    case MatrixOpType::RMS_NORM:
    {
        auto& scale = *in_[1];
        MatrixEx::rmsNormBackward(*in_[0], Y, scale, a_[0]);
        if (in_[1]->needBack())
        {
            //ws[1] = dscale
            Matrix::add(Y.getWorkspace(1), in_[1]->d(), in_[1]->d(), 1, in_[1]->keepWeight());
        }
        break;
    }
    case MatrixOpType::PERMUTE:
    {
        if (in_[0]->needBack())
        {
            auto& perm = anys_[0].to<std::vector<int>>();
            MatrixEx::permute4dBackward(*in_[0], Y, perm);
        }
        break;
    }
    case MatrixOpType::ROPE:
    {
        if (in_[0]->needBack())
        {
            auto& cos_tab = *in_[1];
            auto& sin_tab = *in_[2];
            MatrixEx::ropeBackward(*in_[0], Y, cos_tab, sin_tab);
        }
        //cos / sin 视为常量, 不回传梯度
        break;
    }
    case MatrixOpType::ROPE_INTERLEAVED:
        //推理算子, 暂不实现反向
        break;
    case MatrixOpType::PIXEL_SHUFFLE:
    {
        if (in_[0]->needBack())
        {
            int r = window_[0];
            MatrixEx::pixelShuffleBackward(*in_[0], Y, r);
        }
        break;
    }
    case MatrixOpType::KV_CACHE:
        //推理算子, 不反向
        break;
    case MatrixOpType::PRINT_RATIO:
        //纯诊断, 无反向
        break;
    case MatrixOpType::PRINT_MESSAGE:
        //纯诊断, 无反向
        break;
    case MatrixOpType::DEBUG_SAVE:
        //纯诊断, 无反向
        break;
    case MatrixOpType::ATTENTION:
    {
        float dk = anys_[0].to<float>();
        int causal = (anys_.size() > 1) ? anys_[1].to<int>() : 0;
        MatrixEx::attentionBackward(*in_[0], *in_[1], *in_[2], Y, dk, causal);
        break;
    }
    case MatrixOpType::EMBED:
    {
        // ids 无梯度; W 做 scatter-add
        if (in_[1]->needBack())
        {
            MatrixEx::embedBackward(*in_[0], *in_[1], Y);
        }
        break;
    }
    case MatrixOpType::TILE:
    {
        // in_[0]=X, 反向 scatter-add
        if (in_[0]->needBack())
        {
            MatrixEx::tileBackward(*in_[0], Y, window_);
        }
        break;
    }
    case MatrixOpType::DECONV:
        if (in_[0]->needBack())
        {
            MatrixEx::deconvolutionBackwardDA(*in_[0], *in_[1], Y, stride_, padding_, a_[0], in_[0]->keepWeight());
        }
        if (in_[1]->needBack())
        {
            MatrixEx::deconvolutionBackwardDW(*in_[0], *in_[1], Y, stride_, padding_, a_[0], in_[1]->keepWeight());
        }
        break;
    case MatrixOpType::GROUP_NORM:
    {
        auto& scale = *in_[1];
        auto& bias = *in_[2];
        int G = (int)anys_[0].to<int>();
        if (in_[0]->needBack())
        {
            MatrixEx::groupNormBackward(*in_[0], Y, scale, bias, G, a_[0]);
        }
        if (in_[1]->needBack())
        {
            Matrix::add(Y.getWorkspace(3), in_[1]->d(), in_[1]->d(), 1, in_[1]->keepWeight());
        }
        if (in_[2]->needBack())
        {
            Matrix::add(Y.getWorkspace(4), in_[2]->d(), in_[2]->d(), 1, in_[2]->keepWeight());
        }
        break;
    }
    case MatrixOpType::REPARAM:
        MatrixEx::reparamBackward(*in_[0], *in_[1], Y);
        break;
    case MatrixOpType::UPSAMPLE:
        if (in_[0]->needBack())
        {
            int sh = window_[0], sw = window_[1];
            bool bilinear = (int)anys_[0].to<int>() != 0;
            MatrixEx::upsampleBackward(*in_[0], Y, sh, sw, bilinear);
        }
        break;
    case MatrixOpType::CHUNK:
        if (in_[0]->needBack())
        {
            int start_w = window_[0], size_w = window_[1];
            MatrixEx::chunkBackward(*in_[0], Y, start_w, size_w);
        }
        break;
    case MatrixOpType::SIN_TIME_EMBED:
        // t 视为常量输入, 不回传梯度
        break;
    }
    for (auto& m : in_)
    {
        m->setKeepWeight(1);
    }
}

void MatrixOp::backwardLoss()
{
    //若反向过程需更新多个矩阵，则在函数内部判断needUpdate
    //scale不合理，待计算
    switch (type_)
    {
    case MatrixOpType::LOSS:
        if (scale_ != 0)
        {
            if (in_.size() >= 3 && in_[2]->getDataSize() == in_[0]->getDataSize())
            {
                //有损失权重的情况
                //注意这里使用in_[2]->d()作为了中间变量
                Matrix::add(*in_[0], *in_[1], in_[2]->d(), a_[0] * scale_, -a_[1] * scale_, 0);
                Matrix::elementMul(in_[2]->d(), *in_[2], in_[0]->d(), 1, in_[0]->keepWeight());
                //in_[0]->message("X");
                //in_[2]->message("Y1");
                //in_[0]->d().message("Xd");
            }
            else
            {
                //没有损失权重的情况，也是一般的情况
                //此处直接相减，表示欧氏距离平方，若配合前一层的softmax_ce或sigmoid_ce则表示交叉熵
                Matrix::add(*in_[0], *in_[1], in_[0]->d(), a_[0] * scale_, -a_[1] * scale_, in_[0]->keepWeight());
                //in_[0]->message("X");
                //in_[1]->message("Y1");
                //in_[0]->d().message("Xd");
            }
        }
        break;
    case MatrixOpType::FOCAL:
        if (scale_ != 0)
        {
            if (in_.size() >= 3 && in_[2]->getDataSize() == in_[0]->getDataSize())
            {
                Matrix::add(*in_[0], *in_[1], in_[1]->d(), scale_, -scale_, 0);
                MatrixEx::elementPow(in_[1]->d(), in_[2]->d(), 0.2);
                Matrix::elementMul(in_[2]->d(), *in_[2], in_[0]->d(), 1, in_[0]->keepWeight());
            }
            else
            {
                Matrix::add(*in_[0], *in_[1], in_[1]->d(), scale_, -scale_, in_[0]->keepWeight());
                MatrixEx::elementPow(in_[1]->d(), in_[0]->d(), 0.2);
            }
            //in_[0]->d().message("Xdpow");
        }
        break;
    case MatrixOpType::ZERO_LIMIT:
        if (scale_ != 0)
        {
            MatrixEx::zero_limit(*in_[0], *in_[1], in_[0]->d(), 0.5, 0);
        }
        break;
    case MatrixOpType::L2:
        if (scale_ != 0)
        {
            //unfinished: L2正则化反向未实现
            //Matrix::add(in_[0]->d(), *in_[0], in_[0]->d(), data_weight_, scale_);
        }
        break;
    case MatrixOpType::MSE_LOSS:
        //mean((A-Y)^2), 梯度 = 2*(A-Y)/N * scale_
        if (scale_ != 0)
        {
            float N = (float)in_[0]->getDataSize();
            float alpha = 2.0f * (float)scale_ / N;
            if (in_.size() >= 3 && in_[2]->getDataSize() == in_[0]->getDataSize())
            {
                Matrix::add(*in_[0], *in_[1], in_[2]->d(), alpha, -alpha, 0);
                Matrix::elementMul(in_[2]->d(), *in_[2], in_[0]->d(), 1, in_[0]->keepWeight());
            }
            else
            {
                Matrix::add(*in_[0], *in_[1], in_[0]->d(), alpha, -alpha, in_[0]->keepWeight());
            }
        }
        break;
    case MatrixOpType::L1_LOSS:
        //mean(|A-Y|), 梯度 = sign(A-Y)/N * scale_
        if (scale_ != 0)
        {
            float N = (float)in_[0]->getDataSize();
            float alpha = (float)scale_ / N;
            MatrixEx::l1LossBackward(*in_[0], *in_[1], alpha, in_[0]->keepWeight());
        }
        break;
    case MatrixOpType::KL_LOSS:
        //-0.5*mean(1+log_var-mu^2-exp(log_var)), 先验 N(0,1)
        //d_mu = mu/N * scale_; d_lv = 0.5*(exp(lv)-1)/N * scale_
        if (scale_ != 0)
        {
            float N = (float)in_[0]->getDataSize();
            float alpha = (float)scale_ / N;
            // d_mu[i] = keepWeight*d_mu[i] + alpha*mu[i]
            Matrix::add(*in_[0], *in_[0], in_[0]->d(), alpha, 0.0f, in_[0]->keepWeight());
            // d_lv[i] = keepWeight*d_lv[i] + alpha*0.5*(exp(lv[i])-1)
            MatrixEx::klLvBackward(*in_[1], alpha, in_[1]->keepWeight());
            in_[1]->setKeepWeight(1);
        }
        break;
    }
    in_[0]->setKeepWeight(1);
}

std::string MatrixOp::inference_ir(const std::vector<MatrixOp>& op_queue)
{
    std::string content;
    for (const auto& op : op_queue)
    {
        if (op.connect_a_)
        {
            content += op.print();    //仅用于推理，故有些连接loss，但不连接A的可以不计算
        }
    }
    return content;
}

std::string MatrixOp::print() const
{
    Option op;
    std::string line;
    switch (type_)
    {
    case MatrixOpType::SCALE:
        line = std::format("{} = scale({}, {});", out_[0], in_[0], a_[0]);
        break;
    case MatrixOpType::ADD:
        line = std::format("{} = add({}, {});", out_[0], in_, a_);
        break;
    case MatrixOpType::MUL:
        line = std::format("{} = mul({}, {}, [{}, {}, {}, batch]);", out_[0], in_[0], in_[1],
            out_[0]->getWidth(), out_[0]->getHeight(), out_[0]->getChannel());
        break;
    case MatrixOpType::BATCHED_MUL:
    {
        int ta = (anys_[0].to<MatrixTransType>() == MATRIX_TRANS) ? 1 : 0;
        int tb = (anys_[1].to<MatrixTransType>() == MATRIX_TRANS) ? 1 : 0;
        line = std::format("{} = batchedMul({}, {}, {}, {});",
            out_[0], in_[0], in_[1], ta, tb);
        break;
    }
    case MatrixOpType::ELE_MUL:
        line = std::format("{} = elementMul({}, {}, {});", out_[0], in_[0], in_[1], a_[0]);
        break;
    case MatrixOpType::ADD_BIAS:
        line = std::format("{} = addBias({}, {});", out_[0], in_[0], in_[1]);
        break;
    case MatrixOpType::CONCAT:
        line = std::format("{} = concat({});", out_[0], in_);
        break;
    case MatrixOpType::ACTIVE:
        line = std::format("{} = active({}, active_{}, {}, {});", out_[0], in_[0], op.getStringFromEnum(anys_[0].to<ActiveFunctionType>()), anys_[1].to<std::vector<int>>(), anys_[2].to<std::vector<float>>());
        break;
    case MatrixOpType::POOL:
        if (anys_[1].to<PoolingReverseType>() == POOLING_NOT_REVERSE)
        {
            line = std::format("{} = pool({}, pool_{}, {}, {}, {});", out_[0], in_[0], op.getStringFromEnum(anys_[0].to<PoolingType>()), window_, stride_, padding_);
        }
        else
        {
            line = std::format("{} = reversepool({}, {}, {}, {});", out_[0], in_[0], window_, stride_, padding_);
        }
        break;
    case MatrixOpType::CONV:
        line = std::format("{} = conv({}, {}, {}, {});", out_[0], in_[0], in_[1], stride_, padding_);
        break;
    case MatrixOpType::CORR:
        line = std::format("{} = corr({}, {}, {}, {});", out_[0], in_[0], in_[1], stride_, padding_);
        break;
    case MatrixOpType::RESHAPE:
        line = std::format("{} = reshape({}, {});", out_[0], in_[0], anys_[0].to<std::vector<int>>());
        break;
    case MatrixOpType::MAX:
        line = std::format("{} = max({}, {});", out_[0], in_[0], in_[1]);
        break;
    case MatrixOpType::POOL_CHANNEL:
        line = std::format("{} = poolchannel({}, pool_{});", out_[0], in_[0], op.getStringFromEnum(anys_[0].to<PoolingType>()));
        break;
    case MatrixOpType::LAYER_NORM:
        line = std::format("{} = layerNorm({}, {}, {});", out_[0], in_[0], in_[1], in_[2]);
        break;
    case MatrixOpType::RMS_NORM:
        line = std::format("{} = rmsNorm({}, {});", out_[0], in_[0], in_[1]);
        break;
    case MatrixOpType::PERMUTE:
        line = std::format("{} = permute({}, {});", out_[0], in_[0], anys_[0].to<std::vector<int>>());
        break;
    case MatrixOpType::ROPE:
        line = std::format("{} = rope({}, {}, {});", out_[0], in_[0], in_[1], in_[2]);
        break;
    case MatrixOpType::ROPE_INTERLEAVED:
        line = std::format("{} = rope_interleaved({}, {}, {});", out_[0], in_[0], in_[1], in_[2]);
        break;
    case MatrixOpType::PIXEL_SHUFFLE:
        line = std::format("{} = pixelShuffle({}, {});", out_[0], in_[0], window_[0]);
        break;
    case MatrixOpType::KV_CACHE:
        line = std::format("{} = kvcache({}, {});", out_[0], in_[0], in_[1]);
        break;
    case MatrixOpType::PRINT_RATIO:
        line = std::format("printRatio({}, {});", in_[0], in_[1]);
        break;
    case MatrixOpType::PRINT_MESSAGE:
    {
        std::string label = anys_.size() > 0 ? anys_[0].to<std::string>() : "";
        line = std::format("printMessage({}, \"{}\");", in_[0], label);
        break;
    }
    case MatrixOpType::DEBUG_SAVE:
    {
        std::string filename = anys_.size() > 0 ? anys_[0].to<std::string>() : "";
        line = std::format("saveBinary({}, \"{}\");", in_[0], filename);
        break;
    }
    case MatrixOpType::ATTENTION:
        line = std::format("{} = attention({}, {}, {}, {});", out_[0], in_[0], in_[1], in_[2], anys_[0].to<float>());
        break;
    case MatrixOpType::LOSS:
        line = std::format("addloss(commonloss({}, {}, {}));", int(type_), in_, a_);
        break;
    default:
        line = std::format("addloss(commonloss({}, {}, {}));", int(type_), in_, a_);
    }
    strfunc::replaceAllSubStringRef(line, "[", "{");
    strfunc::replaceAllSubStringRef(line, "]", "}");
    for (auto& out : out_)
    {
        line += std::format("/*{}: out {}*/;", index_, out->sizeMessage(0));
    }
    return line;
}

ActiveFunctionType MatrixOp::getActiveType() const
{
    if (type_ == MatrixOpType::ACTIVE)
    {
        return anys_[0].to<ActiveFunctionType>();
    }
    return ACTIVE_FUNCTION_NONE;
}

PoolingType MatrixOp::getPoolingType() const
{
    if (type_ == MatrixOpType::POOL && !anys_.empty())
    {
        return anys_[0].to<PoolingType>();
    }
    return POOLING_MAX;
}

//检查连接，并判断哪些是权重
void MatrixOp::checkConnect(std::vector<MatrixOp>& op_queue, Matrix& X, Matrix& A, std::vector<MatrixOp>& losses)
{
    enum ConnectType
    {
        CONNECT_X = 1,
        CONNECT_LOSS = -1,
        CONNECT_A = -2,
    };

    std::unordered_map<Matrix*, int> linkX, linkLoss, linkA;
    std::function<void(Matrix&, ConnectType, std::unordered_map<Matrix*, int>&)> check_connect = [&op_queue, &check_connect](Matrix& M, ConnectType direct, std::unordered_map<Matrix*, int>& link_record)
    {
        if (direct == 0)
        {
            return;
        }
        for (int i = 0; i < op_queue.size(); i++)
        {
            bool* connect = nullptr;
            if (direct == CONNECT_X)
            {
                connect = &op_queue[i].connect_x_;
            }
            else if (direct == CONNECT_A)
            {
                connect = &op_queue[i].connect_a_;
            }
            else if (direct == CONNECT_LOSS)
            {
                connect = &op_queue[i].connect_loss_;
            }

            if (*connect)
            {
                continue;
            }
            auto& op = op_queue[i];
            std::vector<MatrixSP>*v1, *v2;
            if (direct == CONNECT_X)
            {
                v1 = &op.in_;
                v2 = &op.out_;
            }
            else
            {
                v1 = &op.out_;
                v2 = &op.in_;
            }
            for (auto& m : *v1)
            {
                if (m->getDataPtr() == M.getDataPtr())
                {
                    *connect = true;
                    for (auto& m2 : *v2)
                    {
                        link_record[m2.get()] = 1;
                        check_connect(*m2, direct, link_record);
                    }
                    break;
                }
            }
        }
    };

    linkX[&X] = 1;
    check_connect(X, CONNECT_X, linkX);
    linkA[&A] = 1;
    check_connect(A, CONNECT_A, linkA);

    for (auto& loss : losses)
    {
        for (auto& m : loss.in_)
        {
            linkLoss[m.get()] = 1;
            check_connect(*m, CONNECT_LOSS, linkLoss);
        }
    }

    int i = 0;
    for (auto it = op_queue.begin(); it != op_queue.end();)
    {
        if (!it->connect_x_ && !it->connect_loss_)
        {
            it = op_queue.erase(it);
        }
        else
        {
            it->index_ = i++;
            for (auto& in : it->in_)
            {
                if (!linkX.contains(in.get()) && linkLoss.contains(in.get()))
                {
                    in->setIsWeight(true);
                }
            }
            for (auto& out : it->out_)
            {
                out->setIsInput(false);    //非输入矩阵
            }
            ++it;
        }
    }
    //repair sigmoid or softmax at the last layer but no cross entropy
    if (op_queue.back().type_ == MatrixOpType::ACTIVE)
    {
        auto active_type = op_queue.back().anys_[0].to<ActiveFunctionType>();
        if (active_type == ACTIVE_FUNCTION_SIGMOID) { active_type = ACTIVE_FUNCTION_SIGMOID_CE; }
        if (active_type == ACTIVE_FUNCTION_SOFTMAX) { active_type = ACTIVE_FUNCTION_SOFTMAX_CE; }
        if (active_type == ACTIVE_FUNCTION_SOFTMAX_FAST) { active_type = ACTIVE_FUNCTION_SOFTMAX_FAST_CE; }
        if (active_type == ACTIVE_FUNCTION_SOFTMAX_CHANNEL) { active_type = ACTIVE_FUNCTION_SOFTMAX_CHANNEL_CE; }
        op_queue.back().anys_[0].to<ActiveFunctionType>() = active_type;
    }
}

void MatrixOp::getDefaultStridePadding(MatrixOpType type, const std::vector<int>& dim, std::vector<int>& stride, std::vector<int>& padding)
{
    if (type == MatrixOpType::CONV || type == MatrixOpType::DECONV)
    {
        if (stride.size() == 0) { stride.resize(dim.size() - 2, 1); }
        if (padding.size() == 0) { padding.resize(dim.size() - 2, 0); }
    }
    if (type == MatrixOpType::POOL)
    {
        if (stride.size() == 0) { stride = dim; }
        if (padding.size() == 0) { padding.resize(dim.size(), 0); }
    }
}

void MatrixOp::as_scale(const MatrixSP& X, const MatrixSP& Y, float r)
{
    Y->resize(X->getDim());
    set(MatrixOpType::SCALE, { X }, { Y }, { r }, { 0 });
}

void MatrixOp::as_mul(const MatrixSP& X1, const MatrixSP& X2, const MatrixSP& Y, float a, std::vector<int> dim)
{
    //此处可强制reshape，返回可以直接卷积的维度
    if (dim.empty())
    {
        dim = X1->getDim();
        dim.back() = X2->getNumber();
    }
    Y->resize(dim);
    set(MatrixOpType::MUL, { X1, X2 }, { Y }, { a });    //这里注意顺序
    if (X1->getNumber() != X2->getRow())
    {
        LOG_ERR("Error: cannot product!\n");
    }
}

//M/N/K/batch 自动从矩阵维度推导, anys_ 只存 ta/tb
//  ta=NO_TRANS: M=X1.width_, K=X1.height_; ta=TRANS: M=X1.height_, K=X1.width_
//  tb=NO_TRANS: N=X2.height_;              tb=TRANS: N=X2.width_
//  batch = X1.number_
void MatrixOp::as_batchedMul(const MatrixSP& X1, const MatrixSP& X2, const MatrixSP& Y,
    MatrixTransType ta, MatrixTransType tb, float a)
{
    int M = (ta == MATRIX_NO_TRANS) ? X1->getWidth() : X1->getHeight();
    int N = (tb == MATRIX_NO_TRANS) ? X2->getHeight() : X2->getWidth();
    int batch = X1->getNumber();
    Y->resize({ M, N, 1, batch });
    VectorAny pv;
    pv.push_back(ta);
    pv.push_back(tb);
    set(MatrixOpType::BATCHED_MUL, { X1, X2 }, { Y }, { a }, {}, std::move(pv));
}

void MatrixOp::as_elementMul(const MatrixSP& X1, const MatrixSP& X2, const MatrixSP& Y, float a)
{
    Y->resize(X1->getDim());
    set(MatrixOpType::ELE_MUL, { X1, X2 }, { Y }, { a }, { 0 });
}

void MatrixOp::as_add(const MatrixSP& X1, const MatrixSP& X2, const MatrixSP& Y, float a, float b)
{
    Y->resize(X1->getDim());
    set(MatrixOpType::ADD, { X1, X2 }, { Y }, { a, b }, { 0 });
}

void MatrixOp::as_add(const std::vector<MatrixSP>& X_vector, const MatrixSP& Y, std::vector<float> a)
{
    Y->resize(X_vector[0]->getDim());
    set(MatrixOpType::ADD, X_vector, { Y }, std::move(a), { 0 });
}

void MatrixOp::as_addBias(const MatrixSP& X, const MatrixSP& bias, const MatrixSP& Y, float a, float b)
{
    //Matrix as_1(A.getNumber(), 1);
    //as_1.fillData(1);
    //需要注意cudnn自带的只支持到5维，若需更多维可以在这里修改写入op_queue的矩阵的维度
    Y->shareData(*X);    //需注意偏移操作是特殊处理的
    Y->resize(X->getDim());
    set(MatrixOpType::ADD_BIAS, { X, bias }, { Y }, { a }, { b });
}

void MatrixOp::as_concat(const std::vector<MatrixSP>& X_vector, const MatrixSP& Y)
{
    if (X_vector.size() > 0)
    {
        int sum_channel = 0;
        for (auto& X : X_vector)
        {
            sum_channel += X->getChannel();
        }
        auto dim = X_vector[0]->getDim();
        dim[dim.size() - 2] = sum_channel;
        Y->resize(dim);
    }
    set(MatrixOpType::CONCAT, X_vector, { Y });
}

void MatrixOp::as_active(const MatrixSP& X, const MatrixSP& Y, ActiveFunctionType af)
{
    Y->resize(X->getDim());
    set(MatrixOpType::ACTIVE, { X }, { Y }, {}, {}, { af, std::vector<int>(), std::vector<float>() });
}

void MatrixOp::as_active(const MatrixSP& X, const MatrixSP& Y, ActiveFunctionType af, std::vector<int>&& int_vector, std::vector<float>&& real_vector, std::vector<Matrix>&& matrix_vector)
{
    Y->resize(X->getDim());
    set(MatrixOpType::ACTIVE, { X }, { Y }, {}, {}, { af, int_vector, real_vector });
}

void MatrixOp::as_pool(const MatrixSP& X, const MatrixSP& Y, PoolingType pooling_type, PoolingReverseType reverse_type, std::vector<int> window, std::vector<int> stride, std::vector<int> padding, float a)
{
    auto dim = X->getDim();
    getDefaultStridePadding(MatrixOpType::POOL, window, stride, padding);
    if (reverse_type == POOLING_NOT_REVERSE)
    {
        for (int i = 0; i < dim.size() - 2; i++)
        {
            dim[i] = (dim[i] + 2 * padding[i] - window[i]) / stride[i] + 1;
        }
    }
    else
    {
        pooling_type = POOLING_AVERAGE_NOPADDING;
        for (int i = 0; i < dim.size() - 2; i++)
        {
            dim[i] = stride[i] * (dim[i] - 1) + window[i] - 2 * padding[i];
        }
    }
    VectorAny pv = { pooling_type, reverse_type, int(window.size()) };
    Y->resize(dim);
    set(MatrixOpType::POOL, { X }, { Y }, { a }, {}, std::move(pv), std::move(window), std::move(stride), std::move(padding));
}

void MatrixOp::as_conv(const MatrixSP& X, const MatrixSP& W, const MatrixSP& Y, std::vector<int> stride, std::vector<int> padding, int conv_algo, float a /*= 1*/)
{
    auto dim = X->getDim();
    getDefaultStridePadding(MatrixOpType::CONV, W->getDim(), stride, padding);
    std::vector<int> v(9);
    for (int i = 0; i < dim.size() - 2; i++)
    {
        dim[i] = (dim[i] + 2 * padding[i] - W->getDim()[i]) / stride[i] + 1;
    }
    dim[dim.size() - 2] = W->getDim().back();
    Y->resize(dim);
    auto t = MatrixOpType::CONV;
    MatrixEx::ConvMethod method;
    if (conv_algo >= 0)
    {
        //仅推导时固定算法
        method.algo = conv_algo;
        method.math_type = 0;
    }
    Y->user_data<MatrixEx::ConvMethod>() = method;
    //在卷积计算开始之前，会查找最快的算法，v中依次保存了前向、反向数据、反向权重的算法编号、张量参数（mathtype）、组数
    set(t, { X, W }, { Y }, { a }, {}, {}, {}, std::move(stride), std::move(padding));
}

void MatrixOp::as_corr(const MatrixSP& X, const MatrixSP& W, const MatrixSP& Y, std::vector<int> stride, std::vector<int> padding, int conv_algo, float a)
{
    as_conv(X, W, Y, stride, padding, conv_algo, a);
    type_ = MatrixOpType::CORR;
}

void MatrixOp::as_reshape(const MatrixSP& X, const MatrixSP& Y, std::vector<int>& dim)
{
    Y->shareData(*X);
    if (!dim.empty())
    {
        dim.back() = X->getNumber();
    }
    //不处理组数
    Y->resize(dim);
    set(MatrixOpType::RESHAPE, { X }, { Y }, {}, {}, { dim });
}

//允许改 batch 维的 reshape: 总元素数必须一致, 否则报错; 适用于 token 维 (D,T,1,B) <-> (D,1,1,T*B) 这类拍平
void MatrixOp::as_reshape_batch(const MatrixSP& X, const MatrixSP& Y, std::vector<int>& dim)
{
    Y->shareData(*X);
    int64_t total_in = 1;
    for (auto d : X->getDim()) { total_in *= d; }
    int64_t total_out = 1;
    for (auto d : dim) { total_out *= d; }
    if (total_in != total_out)
    {
        LOG_ERR("reshape_batch: total element count mismatch ({} vs {})\n", total_in, total_out);
    }
    Y->resize(dim);
    set(MatrixOpType::RESHAPE, { X }, { Y }, {}, {}, { dim });
}

void MatrixOp::as_max(const MatrixSP& X1, const MatrixSP& X2, const MatrixSP& Y)
{
    Y->resize(X1->getDim());
    set(MatrixOpType::MAX, { X1, X2 }, { Y });
}

//批归一化
//scale: 1×1×C×1的缩放参数（可训练权重）
//内部自动创建bias、running_mean、running_variance等辅助矩阵
void MatrixOp::as_batchNorm(const MatrixSP& X, const MatrixSP& scale, const MatrixSP& Y, BatchNormalizationType bn_type, float epsilon)
{
    Y->resize(X->getDim());
    auto dim = scale->getDim();
    auto dt = X->getDataType();
    auto dev = X->getDeviceType();

    //bias是可训练参数，辅助矩阵在首次forward时惰性初始化到Y->workspace_
    auto bias = std::make_shared<Matrix>(dim, dt, dev);
    bias->fillData(0);

    float exp_aver_factor = 0.1f;
    //a_[0] = exp_aver_factor, a_[1] = epsilon
    set(MatrixOpType::BATCH_NORM, { X, scale, bias }, { Y }, { exp_aver_factor, epsilon }, {}, { bn_type });
}

//Layer Normalization
//scale, bias: 形状 [width_]; 沿 inner=width_ 轴归一化
void MatrixOp::as_layerNorm(const MatrixSP& X, const MatrixSP& scale, const MatrixSP& bias, const MatrixSP& Y, float epsilon)
{
    Y->resize(X->getDim());
    //a_[0] = epsilon
    set(MatrixOpType::LAYER_NORM, { X, scale, bias }, { Y }, { epsilon });
}

//RMS Normalization
void MatrixOp::as_rmsNorm(const MatrixSP& X, const MatrixSP& scale, const MatrixSP& Y, float epsilon)
{
    Y->resize(X->getDim());
    set(MatrixOpType::RMS_NORM, { X, scale }, { Y }, { epsilon });
}

//4D 任意轴置换: dim_out[i] = dim_in[perm[i]]
void MatrixOp::as_permute(const MatrixSP& X, const MatrixSP& Y, const std::vector<int>& perm)
{
    if (perm.size() != 4)
    {
        LOG_ERR("permute: perm size must be 4 (got {})\n", perm.size());
        return;
    }
    auto dim_in = X->getDim();
    while (dim_in.size() < 4) { dim_in.push_back(1); }
    std::vector<int> dim_out(4);
    for (int i = 0; i < 4; i++) { dim_out[i] = dim_in[perm[i]]; }
    Y->resize(dim_out);
    auto perm_copy = perm;
    set(MatrixOpType::PERMUTE, { X }, { Y }, {}, {}, { perm_copy });
}

//RoPE
void MatrixOp::as_rope(const MatrixSP& X, const MatrixSP& cos_tab, const MatrixSP& sin_tab, const MatrixSP& Y)
{
    if (X->getWidth() % 2 != 0)
    {
        LOG_ERR("rope: D (= width_) must be even (got {})\n", X->getWidth());
        return;
    }
    Y->resize(X->getDim());
    set(MatrixOpType::ROPE, { X, cos_tab, sin_tab }, { Y });
}

void MatrixOp::as_rope_interleaved(const MatrixSP& X, const MatrixSP& cos_tab, const MatrixSP& sin_tab, const MatrixSP& Y)
{
    if (X->getWidth() % 2 != 0)
    {
        LOG_ERR("rope_interleaved: D (= width_) must be even (got {})\n", X->getWidth());
        return;
    }
    Y->resize(X->getDim());
    set(MatrixOpType::ROPE_INTERLEAVED, { X, cos_tab, sin_tab }, { Y });
}

//KV cache: 记录到 cache, Y 与 cache 共享内存
void MatrixOp::as_kvcache(const MatrixSP& X_new, const MatrixSP& cache, const MatrixSP& Y)
{
    auto dim_x = X_new->getDim();
    auto dim_c = cache->getDim();
    while (dim_x.size() < 4) { dim_x.push_back(1); }
    while (dim_c.size() < 4) { dim_c.push_back(1); }
    if (dim_x[0] != dim_c[0] || dim_x[2] != dim_c[2] || dim_x[3] != dim_c[3])
    {
        LOG_ERR("kvcache: X_new and cache must match on dims [W, C, N] (X={}, cache={})\n",
            dim_x, dim_c);
        return;
    }
    if (dim_x[1] > dim_c[1])
    {
        LOG_ERR("kvcache: X_new T_new ({}) > cache T_max ({})\n", dim_x[1], dim_c[1]);
        return;
    }
    //Y 与 cache 同 shape 同指针
    Y->resize(cache->getDim());
    Y->shareData(*cache);
    //window_ = { current_pos, T_max }
    set(MatrixOpType::KV_CACHE, { X_new, cache }, { Y }, {}, {}, {}, { 0, dim_c[1] });
}

void MatrixOp::resetKVCache()
{
    if (type_ == MatrixOpType::KV_CACHE && !window_.empty())
    {
        window_[0] = 0;
    }
}

void MatrixOp::resetKVCache(std::vector<MatrixOp>& op_queue)
{
    for (auto& op : op_queue)
    {
        op.resetKVCache();
    }
}

void MatrixOp::as_pixelShuffle(const MatrixSP& X, const MatrixSP& Y, int r)
{
    auto dim = X->getDim();
    while (dim.size() < 4)
    {
        dim.push_back(1);
    }
    int W = dim[0], H = dim[1], C_in = dim[2], N = dim[3];
    if (r <= 0 || C_in % (r * r) != 0)
    {
        LOG_ERR("pixelShuffle: channel ({}) must be divisible by r^2 ({}) and r must be > 0\n", C_in, r * r);
        return;
    }
    int C_out = C_in / (r * r);
    Y->resize({ W * r, H * r, C_out, N });
    set(MatrixOpType::PIXEL_SHUFFLE, { X }, { Y }, {}, {}, {}, { r });
}

void MatrixOp::as_poolChannel(const MatrixSP& X, const MatrixSP& Y, PoolingType pooling_type, PoolingReverseType reverse_type, float a)
{
    auto dim = X->getDim();
    dim[dim.size() - 2] = 1;
    Y->resize(dim);
    set(MatrixOpType::POOL_CHANNEL, { X }, { Y }, { a }, {}, { pooling_type, reverse_type });
}

void MatrixOp::as_prependToken(const MatrixSP& X, const MatrixSP& cls, const MatrixSP& Y)
{
    //X: (D, T, 1, B), cls: (D, 1, 1, 1) -> Y: (D, T+1, 1, B)
    auto dim = X->getDim();
    if (dim.size() < 4)
    {
        LOG_ERR("prependToken expects 4D input (D,T,1,B)\n");
    }
    dim[1] = dim[1] + 1;    //H 维 +1
    Y->resize(dim);
    set(MatrixOpType::PREPEND_TOKEN, { X, cls }, { Y });
}

void MatrixOp::as_firstToken(const MatrixSP& X, const MatrixSP& Y)
{
    //X: (D, T, 1, B) -> Y: (D, 1, 1, B)
    auto dim = X->getDim();
    if (dim.size() < 4)
    {
        LOG_ERR("firstToken expects 4D input (D,T,1,B)\n");
    }
    dim[1] = 1;
    Y->resize(dim);
    set(MatrixOpType::FIRST_TOKEN, { X }, { Y });
}

void MatrixOp::as_print_ratio(const MatrixSP& A, const MatrixSP& B, const MatrixSP& dummy, const std::string& label)
{
    // 纯诊断算子：每隔若干 forward 打印 RMS(A)/RMS(B)，不影响梯度
    dummy->resize(1, 1, 1, 1);
    set(MatrixOpType::PRINT_RATIO, { A, B }, { dummy }, {}, {}, VectorAny{ label });
}

void MatrixOp::as_print_message(const MatrixSP& X, const MatrixSP& dummy, const std::string& label)
{
    // 纯诊断算子：每次 forward 打印 X 的 Dim/L1/L2，不影响梯度
    dummy->resize(1, 1, 1, 1);
    set(MatrixOpType::PRINT_MESSAGE, { X }, { dummy }, {}, {}, VectorAny{ label });
}

void MatrixOp::as_save_binary(const MatrixSP& X, const MatrixSP& dummy, const std::string& filename)
{
    // 纯诊断算子：首次 forward 将矩阵保存为 float32 binary 文件，不影响梯度
    dummy->resize(1, 1, 1, 1);
    set(MatrixOpType::DEBUG_SAVE, { X }, { dummy }, {}, {}, VectorAny{ filename });
}

void MatrixOp::as_attention(const MatrixSP& Q, const MatrixSP& K, const MatrixSP& V, const MatrixSP& Y, float dk, int causal)
{
    // Scaled Dot-Product Attention: Y = softmax_channel(K^T @ Q / sqrt(dk)) @ V
    // Q/K/V/Y 形状均为 (D, T, 1, B); causal=1 时应用因果掩码
    int D = Q->getWidth(), T = Q->getHeight(), B = Q->getNumber();
    Y->resize({ D, T, 1, B });
    VectorAny pv;
    pv.push_back(dk);
    pv.push_back(causal);
    set(MatrixOpType::ATTENTION, { Q, K, V }, { Y }, {}, {}, std::move(pv));
}

void MatrixOp::as_embed(const MatrixSP& ids, const MatrixSP& W, const MatrixSP& Y)
{
    // ids: (T,1,1,B) float-as-int, W: (D,1,1,V) -> Y: (D,T,1,B)
    int D = W->getWidth();
    int T = ids->getWidth();
    int B = ids->getNumber();
    Y->resize({ D, T, 1, B });
    // ids 不参与反向（离散）
    ids->setNeedBack(false);
    set(MatrixOpType::EMBED, { ids, W }, { Y });
}

void MatrixOp::as_tile(const MatrixSP& X, const MatrixSP& Y, const std::vector<int>& repeats)
{
    // Y 形状 = X 形状各维 × repeats[i]
    std::vector<int> r = repeats;
    r.resize(4, 1);    // 补齐到 4 个维度
    std::vector<int> out_dim = {
        X->getWidth() * r[0],
        X->getHeight() * r[1],
        X->getChannel() * r[2],
        X->getNumber() * r[3]
    };
    Y->resize(out_dim);
    // 把 repeats 存入 window_ (4 元素: r0,r1,r2,r3)
    set(MatrixOpType::TILE, { X }, { Y }, {}, {}, {}, std::vector<int>(r));
}

//转置卷积: A[W_in,H_in,C_in,N] × W[kW,kH,C_out,C_in] -> Y[W_out,H_out,C_out,N]
//C_in = W->getNumber(), C_out = W->getChannel()
//Y_W = (A_W - 1)*stride_W - 2*padding_W + kW
void MatrixOp::as_deconv(const MatrixSP& A, const MatrixSP& W, const MatrixSP& Y,
    std::vector<int> stride, std::vector<int> padding, int conv_algo, float a)
{
    getDefaultStridePadding(MatrixOpType::DECONV, W->getDim(), stride, padding);
    auto dim = A->getDim();
    // 输出空间尺寸: (dim[i] - 1)*stride[i] - 2*padding[i] + W.dim[i]
    for (int i = 0; i < (int)dim.size() - 2; i++)
    {
        dim[i] = (dim[i] - 1) * stride[i] - 2 * padding[i] + W->getDim()[i];
    }
    // C_out = W->getChannel() (deconv 输出通道)
    dim[(int)dim.size() - 2] = W->getChannel();
    Y->resize(dim);
    auto t = MatrixOpType::DECONV;
    MatrixEx::ConvMethod method;
    if (conv_algo >= 0)
    {
        method.algo = conv_algo;
    }
    Y->user_data<MatrixEx::ConvMethod>() = method;
    set(t, { A, W }, { Y }, { a }, {}, {}, {}, std::move(stride), std::move(padding));
}

//Group Normalization
void MatrixOp::as_groupNorm(const MatrixSP& X, const MatrixSP& scale, const MatrixSP& bias,
    const MatrixSP& Y, int G, float epsilon)
{
    Y->resize(X->getDim());
    // scale/bias 初始化为 1/0
    scale->fillData(1);
    bias->fillData(0);
    // a_[0] = epsilon; anys_[0] = G
    set(MatrixOpType::GROUP_NORM, { X, scale, bias }, { Y }, { epsilon }, {}, { G });
}

//VAE 重参数化
void MatrixOp::as_reparam(const MatrixSP& mu, const MatrixSP& log_var, const MatrixSP& z)
{
    z->resize(mu->getDim());
    set(MatrixOpType::REPARAM, { mu, log_var }, { z });
}

//Nearest/Bilinear upsample: X[W,H,C,N] -> Y[W*sw, H*sh, C, N]
void MatrixOp::as_upsample(const MatrixSP& X, const MatrixSP& Y, int sh, int sw, bool bilinear)
{
    auto dim = X->getDim();
    dim[0] = dim[0] * sw;    // W*sw
    dim[1] = dim[1] * sh;    // H*sh
    Y->resize(dim);
    VectorAny pv;
    pv.push_back((int)bilinear);
    set(MatrixOpType::UPSAMPLE, { X }, { Y }, {}, {}, std::move(pv), { sh, sw });
}

// Chunk: 沿 width(axis=0) 取第 chunk_i 块 (共 n_total 块)
void MatrixOp::as_chunk(const MatrixSP& X, const MatrixSP& Y, int chunk_i, int n_total)
{
    int W = X->getWidth();
    int size_w = W / n_total;
    int start_w = chunk_i * size_w;
    auto dim = X->getDim();
    dim[0] = size_w;
    Y->resize(dim);
    set(MatrixOpType::CHUNK, { X }, { Y }, {}, {}, {}, { start_w, size_w });
}

// SliceW: 沿 width(axis=0) 取任意范围 [start_w, start_w+size_w)
void MatrixOp::as_sliceW(const MatrixSP& X, const MatrixSP& Y, int start_w, int size_w)
{
    auto dim = X->getDim();
    dim[0] = size_w;
    Y->resize(dim);
    set(MatrixOpType::CHUNK, { X }, { Y }, {}, {}, {}, { start_w, size_w });
}

// 正弦时间步嵌入: t (1,1,1,B) -> emb (d,1,1,B)
void MatrixOp::as_sinTimeEmbed(const MatrixSP& t, const MatrixSP& Y, int d, float base)
{
    int B = t->getNumber();
    Y->resize({ d, 1, 1, B });
    set(MatrixOpType::SIN_TIME_EMBED, { t }, { Y }, { base }, {}, {}, { d });
}

std::vector<MatrixOp> operator+(const std::vector<MatrixOp>& A, const std::vector<MatrixOp>& B)
{
    auto R = A;
    R.insert(R.end(), B.begin(), B.end());
    return R;
}

std::vector<MatrixOp>& operator+=(std::vector<MatrixOp>& A, const std::vector<MatrixOp>& B)
{
    A.insert(A.end(), B.begin(), B.end());
    return A;
}

std::vector<MatrixOp> operator*(const std::vector<MatrixOp>& A, double v)
{
    auto R = A;
    for (auto& R1 : R)
    {
        R1.scale_ *= v;
    }
    return R;
}

std::vector<MatrixOp> operator*(double v, const std::vector<MatrixOp>& A)
{
    auto R = A;
    for (auto& R1 : R)
    {
        R1.scale_ *= v;
    }
    return R;
}

std::vector<MatrixOp> commonLoss(MatrixOpType type, const std::vector<MatrixSP>& M, const std::vector<float>& a)
{
    MatrixOp op;
    auto a1 = a;
    op.set(type, M, {}, std::move(a1), {});
    return { op };
}

std::vector<MatrixOp> crossEntropy(const MatrixSP& A, const MatrixSP& Y)
{
    MatrixOp op;
    op.set(MatrixOpType::LOSS, { A, Y }, {}, {});
    return { op };
}

std::vector<MatrixOp> crossEntropy(const MatrixSP& A, const MatrixSP& Y, const MatrixSP& LW)
{
    MatrixOp op;
    op.set(MatrixOpType::LOSS, { A, Y, LW }, {}, {});
    return { op };
}

std::vector<MatrixOp> focal(const MatrixSP& A, const MatrixSP& Y)
{
    MatrixOp op;
    op.set(MatrixOpType::FOCAL, { A, Y }, {}, {});
    return { op };
}

std::vector<MatrixOp> focal(const MatrixSP& A, const MatrixSP& Y, const MatrixSP& LW)
{
    MatrixOp op;
    op.set(MatrixOpType::FOCAL, { A, Y, LW }, {}, {});
    return { op };
}

std::vector<MatrixOp> L2(const MatrixSP& A)
{
    MatrixOp op;
    op.set(MatrixOpType::L2, { A }, {}, {});
    return { op };
}

std::vector<MatrixOp> L2(const std::vector<MatrixSP>& v)
{
    std::vector<MatrixOp> q;
    for (auto& m : v)
    {
        MatrixOp op;
        op.set(MatrixOpType::L2, { m }, {}, {});
        q.push_back(op);
    }
    return q;
}

std::vector<MatrixOp> mseLoss(const MatrixSP& A, const MatrixSP& Y)
{
    MatrixOp op;
    op.set(MatrixOpType::MSE_LOSS, { A, Y }, {}, {});
    return { op };
}

std::vector<MatrixOp> l1Loss(const MatrixSP& A, const MatrixSP& Y)
{
    MatrixOp op;
    op.set(MatrixOpType::L1_LOSS, { A, Y }, {}, {});
    return { op };
}

std::vector<MatrixOp> klLoss(const MatrixSP& mu, const MatrixSP& log_var)
{
    MatrixOp op;
    op.set(MatrixOpType::KL_LOSS, { mu, log_var }, {}, {});
    return { op };
}

}    // namespace cccc