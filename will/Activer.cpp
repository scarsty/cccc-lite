#include "Activer.h"

namespace cccc
{

Activer::Activer()
{
}

Activer::~Activer()
{
}

void Activer::init(Option* op, std::string section, LayerConnectionType ct)
{
    active_function_ = op->getEnum(section, "active", ACTIVE_FUNCTION_NONE);
    cost_function_ = op->getEnum2(section, "cost", COST_FUNCTION_CROSS_ENTROPY);    //cost似乎是应该属于net的，待查
    cost_rate_ = op->getVector<real>(section, "cost_rate");

    real learn_rate_base = op->getReal2(section, "learn_rate_base", 1e-2);
    switch (active_function_)
    {
    case ACTIVE_FUNCTION_CLIPPED_RELU:
        real_vector_ = { op->getReal2(section, "clipped_relu", 0.5) };
        break;
    case ACTIVE_FUNCTION_DROPOUT:
    {
        //Dropout会重设两个矩阵
        //2 int: work_phase, seed
        int_vector_ =
        {
            int(ACTIVE_PHASE_TRAIN),
            int(op->getReal2(section, "dropout_seed", INT_MAX * random_generator_.rand()))
        };
        random_generator_.set_seed();
        random_generator_.set_random_type(RANDOM_UNIFORM);
        random_generator_.set_parameter(0, 1);
        //1 real: dropout_rate
        real_vector_ =
        {
            op->getReal2(section, "dropout_rate", 0.5)
        };
        break;
    }
    case ACTIVE_FUNCTION_LOCAL_RESPONSE_NORMALIZATION:
        //1 int: lrn_n
        int_vector_ =
        {
            op->getInt2(section, "lrn_n", 5)
        };
        //3 real: lrn_alpha, lrn_beta, lrn_k
        real_vector_ =
        {
            op->getReal2(section, "lrn_alpha", 1e-4),
            op->getReal2(section, "lrn_beta", 0.75),
            op->getReal2(section, "lrn_k", 2.0)
        };
        break;
    case ACTIVE_FUNCTION_LOCAL_CONSTRAST_NORMALIZATION:
        //1 int: lcn_n
        int_vector_ =
        {
            op->getInt2(section, "lcn_n", 5)
        };
        //3 real: lcn_alpha, lcn_beta, lcn_k (not sure these are useful)
        real_vector_ =
        {
            op->getReal2(section, "lcn_alpha", 1),
            op->getReal2(section, "lcn_beta", 0.5),
            op->getReal2(section, "lcn_k", 1e-5)
        };
        break;
    case ACTIVE_FUNCTION_DIVISIVE_NORMALIZATION:
        int_vector_ =
        {
            int(op->getReal2(section, "dn_n", 5))
        };
        //same to lcn, real[3] is learn rate of means
        real_vector_ =
        {
            op->getReal2(section, "dn_alpha", 1),
            op->getReal2(section, "dn_beta", 0.5),
            op->getReal2(section, "dn_k", 1e-5),
            op->getReal2(section, "dn_rate", learn_rate_base)
        };
        break;
    case ACTIVE_FUNCTION_BATCH_NORMALIZATION:
    {
        auto bt = op->getEnum(section, "bn_type", BATCH_NORMALIZATION_AUTO);
        if (bt == BATCH_NORMALIZATION_AUTO)
        {
            if (ct == LAYER_CONNECTION_CONVOLUTION)
            {
                bt = BATCH_NORMALIZATION_SPATIAL;
            }
            else
            {
                bt = BATCH_NORMALIZATION_PER_ACTIVATION;
            }
        }
        //2 int: work_phase, batch_normalization
        int_vector_ =
        {
            int(ACTIVE_PHASE_TRAIN),
            int(bt)
        };
        //3 real: train_rate, exp_aver_factor, epsilon
        real_vector_ =
        {
            op->getReal2(section, "bn_rate", learn_rate_base),
            op->getReal2(section, "bn_exp_aver_factor", 1),
            op->getReal2(section, "bn_epsilon", 1e-5)
        };
        break;
    }
    case ACTIVE_FUNCTION_SPATIAL_TRANSFORMER:
        //1 real: train_rate
        real_vector_ =
        {
            op->getReal2(section, "st_train_rate", 0)
        };
        break;
    case ACTIVE_FUNCTION_RECURRENT:
        int_vector_ =
        {
            op->getInt(section, "rnn_num_layers", 2),
            op->getEnum(section, "rnn_type", RECURRENT_RELU),
            op->getEnum(section, "rnn_direction", RECURRENT_DIRECTION_UNI),
            op->getEnum(section, "rnn_input", RECURRENT_INPUT_LINEAR)
        };
        break;
    case ACTIVE_FUNCTION_ZERO_CHANNEL:
        int_vector_ = op->getVector<int>(section, "zero_channels");
        break;
    default:
        break;
    }
}

//激活前的准备工作，主要用于一些激活函数在训练和测试时有不同设置
void Activer::activePrepare(ActivePhaseType ap)
{
    switch (active_function_)
    {
    case ACTIVE_FUNCTION_DROPOUT:
        int_vector_[0] = int(ap);
        if (ap == ACTIVE_PHASE_TRAIN)
        {
            random_generator_.set_random_type(RANDOM_UNIFORM);
            int_vector_[1] = int(INT_MAX * random_generator_.rand());
        }
        break;
    case ACTIVE_FUNCTION_BATCH_NORMALIZATION:
        int_vector_[0] = int(ap);
        break;
    default:
        break;
    }
}

void Activer::initBuffer(Matrix& X)
{
    MatrixEx::activeBufferInit(X, active_function_, int_vector_, real_vector_, matrix_vector_);
    if (!cost_rate_.empty())
    {
        matrix_cost_rate_ = Matrix(X.getDim(), DeviceType::CPU);
        matrix_cost_rate_.initData(1);
        for (int ir = 0; ir < X.getRow(); ir++)
        {
            for (int in = 0; in < X.getNumber(); in++)
            {
                if (ir < cost_rate_.size())
                {
                    matrix_cost_rate_.getData(ir, in) = cost_rate_[ir];
                }
            }
        }
        matrix_cost_rate_.toGPU();
    }
}

void Activer::backwardCost(Matrix& A, Matrix& X, Matrix& Y, Matrix& dA, Matrix& dX)
{
    if (cost_function_ == COST_FUNCTION_RMSE)
    {
        Matrix::add(A, Y, dA, 1, -1);
        backward(A, X);
    }
    else
    {
        //交叉熵的推导结果就是直接相减，注意此处并未更新DA
        Matrix::add(A, Y, dX, 1, -1);
    }
    if (matrix_cost_rate_.getDataSize() > 0)
    {
        Matrix::elementMul(dX, matrix_cost_rate_, dX);
    }
}

//dY will be changed
//实际上反向传播并未直接使用，额外消耗计算资源，不要多用
real Activer::calCostValue(Matrix& A, Matrix& dA, Matrix& Y, Matrix& dY)
{
    //todo::seem not right
    if (dY.getDataSize() == 0)
    {
        return 0;
    }
    if (cost_function_ == COST_FUNCTION_RMSE)
    {
        Matrix::add(A, Y, dY, 1, -1);
        return dY.dotSelf();
    }
    else
    {
        switch (active_function_)
        {
        case ACTIVE_FUNCTION_SIGMOID:
            Matrix::crossEntropy2(A, Y, dY, 1e-5);
            break;
        case ACTIVE_FUNCTION_SOFTMAX:
        case ACTIVE_FUNCTION_SOFTMAX_FAST:
            Matrix::crossEntropy(A, Y, dY, 1e-5);
            break;
        default:
            Matrix::add(A, Y, dY, 1, -1);
            break;
        }
        return dY.sumAbs();
    }
}

bool Activer::needSave(int i)
{
    return active_function_ == ACTIVE_FUNCTION_BATCH_NORMALIZATION
        || active_function_ == ACTIVE_FUNCTION_SPATIAL_TRANSFORMER;
}

}    // namespace cccc