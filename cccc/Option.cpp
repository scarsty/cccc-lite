#include "Option.h"
#include <algorithm>

namespace cccc
{

Option::Option()
{
    initEnums();
}

Option::Option(const std::string& filename) : Option()
{
    loadFile(filename);
}

void Option::setKeys(const std::string& section, const std::string& pairs)
{
    setKeys(section, strfunc::splitString(pairs, ";"));
}

void Option::setKeys(const std::string& section, const std::vector<std::string>& pairs)
{
    for (auto pair : pairs)
    {
        strfunc::replaceAllSubStringRef(pair, " ", "");
        auto p = pair.find("=");
        if (p != std::string::npos)
        {
            auto key = pair.substr(0, p);
            auto value = pair.substr(p + 1);
            setKey(section, key, value);
        }
    }
}

std::string Option::dealString(std::string str, int to_filename /*= 0*/)
{
    size_t p = 0;
    while ((p = str.find("{", p)) != std::string::npos)
    {
        auto p0 = p;
        p++;
        std::string section, key;
        auto p1 = str.find("::", p);
        if (p1 != std::string::npos)
        {
            section = str.substr(p, p1 - p);
            p = p1 + 2;
        }
        auto p2 = str.find("}", p);
        {
            key = str.substr(p, p2 - p);
            p = p2 + 1;
        }
        if (hasKey(section, key))
        {
            std::string sub = getString(section, key);
            if (to_filename)
            {
                strfunc::replaceAllSubStringRef(sub, ":", "");
                strfunc::replaceAllSubStringRef(sub, "\\", "");
                strfunc::replaceAllSubStringRef(sub, "/", "");
            }
            strfunc::replaceAllSubStringRef(str, str.substr(p0, p - p0), sub);
            p = p0 + sub.size();
        }
    }
    return str;
}

//初始化map，注意一些设置是有别名的
//所有字符小写！！！！
//第一项通常是默认项
void Option::initEnums()
{
    registerEnum<ActiveFunctionType>(
        {
            { "none", ACTIVE_FUNCTION_NONE },
            { "sigmoid", ACTIVE_FUNCTION_SIGMOID },
            { "relu", ACTIVE_FUNCTION_RELU },
            { "tanh", ACTIVE_FUNCTION_TANH },
            { "clipped_relu", ACTIVE_FUNCTION_CLIPPED_RELU },
            { "elu", ACTIVE_FUNCTION_ELU },
            { "softmax", ACTIVE_FUNCTION_SOFTMAX },
            { "softmax_fast", ACTIVE_FUNCTION_SOFTMAX_FAST },
            { "softmax_log", ACTIVE_FUNCTION_SOFTMAX_LOG },
            { "dropout", ACTIVE_FUNCTION_DROPOUT },
            { "recurrent", ACTIVE_FUNCTION_RECURRENT },
            { "softplus", ACTIVE_FUNCTION_SOFTPLUS },
            { "local_response_normalization", ACTIVE_FUNCTION_LOCAL_RESPONSE_NORMALIZATION },
            { "lrn", ACTIVE_FUNCTION_LOCAL_RESPONSE_NORMALIZATION },
            { "local_constrast_normalization", ACTIVE_FUNCTION_LOCAL_CONSTRAST_NORMALIZATION },
            { "lcn", ACTIVE_FUNCTION_LOCAL_CONSTRAST_NORMALIZATION },
            { "divisive_normalization", ACTIVE_FUNCTION_DIVISIVE_NORMALIZATION },
            { "dn", ACTIVE_FUNCTION_DIVISIVE_NORMALIZATION },
            { "batch_normalization", ACTIVE_FUNCTION_BATCH_NORMALIZATION },
            { "bn", ACTIVE_FUNCTION_BATCH_NORMALIZATION },
            { "spatial_transformer", ACTIVE_FUNCTION_SPATIAL_TRANSFORMER },
            { "sum_max", ACTIVE_FUNCTION_SUMMAX },
            { "zero_channel", ACTIVE_FUNCTION_ZERO_CHANNEL },
            { "sigmoid_ce", ACTIVE_FUNCTION_SIGMOID_CE },
            { "softmax_ce", ACTIVE_FUNCTION_SOFTMAX_CE },
            { "softmax_fast_ce", ACTIVE_FUNCTION_SOFTMAX_FAST_CE },
            { "sin", ACTIVE_FUNCTION_SIN },
            { "zigzag", ACTIVE_FUNCTION_ZIGZAG },
            { "leaky_relu", ACTIVE_FUNCTION_LEAKY_RELU },
            { "lrelu", ACTIVE_FUNCTION_LEAKY_RELU },
            { "selu", ACTIVE_FUNCTION_SELU },
            { "square", ACTIVE_FUNCTION_SQUARE },
            { "abs", ACTIVE_FUNCTION_ABS },
            { "sinp", ACTIVE_FUNCTION_SIN_PLUS },
            { "silu", ACTIVE_FUNCTION_SILU },
        });

    registerEnum<LayerConnectionType>(
        {
            { "none", LAYER_CONNECTION_NONE },
            { "null", LAYER_CONNECTION_NONE },
            { "full_connect", LAYER_CONNECTION_FULLCONNECT },
            { "full", LAYER_CONNECTION_FULLCONNECT },
            { "fc", LAYER_CONNECTION_FULLCONNECT },
            { "convolution", LAYER_CONNECTION_CONVOLUTION },
            { "conv", LAYER_CONNECTION_CONVOLUTION },
            { "pooling", LAYER_CONNECTION_POOLING },
            { "pool", LAYER_CONNECTION_POOLING },
            { "direct", LAYER_CONNECTION_DIRECT },
            { "correlation", LAYER_CONNECTION_CORRELATION },
            { "corr", LAYER_CONNECTION_CORRELATION },
            { "conv2", LAYER_CONNECTION_CORRELATION },
            { "combine", LAYER_CONNECTION_COMBINE },
            { "extract", LAYER_CONNECTION_EXTRACT },
            { "roteigen", LAYER_CONNECTION_ROTATE_EIGEN },
            { "norm2", LAYER_CONNECTION_NORM2 },
            { "transpose", LAYER_CONNECTION_TRANSPOSE },
            { "nac", LAYER_CONNECTION_NAC },
        });

    registerEnum<CostFunctionType>(
        {
            { "rmse", COST_FUNCTION_RMSE },
            { "cross_entropy", COST_FUNCTION_CROSS_ENTROPY },
        });

    registerEnum<PoolingType>(
        {
            { "max", POOLING_MAX },
            { "average", POOLING_AVERAGE_NOPADDING },
            { "average_no_padding", POOLING_AVERAGE_NOPADDING },
            { "average_padding", POOLING_AVERAGE_PADDING },
        });

    registerEnum<CombineType>(
        {
            { "concat", COMBINE_CONCAT },
            { "add", COMBINE_ADD },
        });

    registerEnum<RandomFillType>(
        {
            { "xavier", RANDOM_FILL_XAVIER },
            { "constant", RANDOM_FILL_CONSTANT },
            { "gaussian", RANDOM_FILL_GAUSSIAN },
            { "msra", RANDOM_FILL_GAUSSIAN },
            { "lecun", RANDOM_FILL_LECUN },
        });

    registerEnum<AdjustLearnRateType>(
        {
            { "fixed", ADJUST_LEARN_RATE_FIXED },
            { "scale_inter", ADJUST_LEARN_RATE_SCALE_INTER },
            { "linear_inter", ADJUST_LEARN_RATE_LINEAR_INTER },
        });

    registerEnum<BatchNormalizationType>(
        {
            { "per_active", BATCH_NORMALIZATION_PER_ACTIVATION },
            { "spatial", BATCH_NORMALIZATION_SPATIAL },
            { "auto", BATCH_NORMALIZATION_AUTO },
        });

    registerEnum<RecurrentType>(
        {
            { "relu", RECURRENT_RELU },
            { "tanh", RECURRENT_TANH },
            { "lstm", RECURRENT_LSTM },
            { "gru", RECURRENT_GRU },
        });

    registerEnum<RecurrentDirectionType>(
        {
            { "uni", RECURRENT_DIRECTION_UNI },
            { "bi", RECURRENT_DIRECTION_BI },
        });

    registerEnum<RecurrentInputType>(
        {
            { "linear", RECURRENT_INPUT_LINEAR },
            { "skip", RECURRENT_INPUT_SKIP },
        });

    registerEnum<RecurrentAlgoType>(
        {
            { "standard", RECURRENT_ALGO_STANDARD },
            { "static", RECURRENT_ALGO_PERSIST_STATIC },
            { "dynamic", RECURRENT_ALGO_PERSIST_DYNAMIC },
        });

    registerEnum<SolverType>(
        {
            { "sgd", SOLVER_SGD },
            { "nag", SOLVER_NAG },
            { "ada_delta", SOLVER_ADA_DELTA },
            { "adam", SOLVER_ADAM },
            { "rms_prop", SOLVER_RMS_PROP },
        });

    registerEnum<WorkModeType>(
        {
            { "normal", WORK_MODE_NORMAL },
            { "gan", WORK_MODE_GAN },
            { "prune", WORK_MODE_PRUNE },
        });

    registerEnum<PruneType>(
        {
            { "active", PRUNE_ACTIVE },
            { "weight", PRUNE_WEIGHT },
        });
}

}    // namespace cccc