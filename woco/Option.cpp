#include "Option.h"
#include <algorithm>

namespace woco
{

Option::Option()
{
    initEnums();
}

Option::Option(const std::string& filename) : Option()
{
    loadIniFile(filename);
}

void Option::loadIniFile(const std::string& filename)
{
    std::string content = convert::readStringFromFile(filename);
    loadIniString(content);
    //print();
}

void Option::loadIniString(const std::string& content)
{
    ini_reader_.loadString(content);
}

int Option::getInt(const std::string& section, const std::string& key, int default_value)
{
    return int(getReal(section, key, default_value));
}

double Option::getReal(const std::string& section, const std::string& key, double default_value)
{
    return ini_reader_.getReal(section, key, default_value);
}

//提取字串属性，会去掉单引号，双引号
std::string Option::getString(const std::string& section, const std::string& key, const std::string& default_value)
{
    std::string str = ini_reader_.getString(section, key, default_value);
    convert::replaceAllSubStringRef(str, "\'", "");
    convert::replaceAllSubStringRef(str, "\"", "");
    return str;
}

void Option::setOptions(const std::string& section, const std::string& pairs)
{
    setOptions(section, convert::splitString(pairs, ";"));
}

void Option::setOptions(const std::string& section, const std::vector<std::string>& pairs)
{
    for (auto pair : pairs)
    {
        convert::replaceAllSubStringRef(pair, " ", "");
        auto p = pair.find("=");
        if (p != std::string::npos)
        {
            auto key = pair.substr(0, p);
            auto value = pair.substr(p + 1);
            setOption(section, key, value);
        }
    }
}

void Option::print()
{
    ini_reader_.print();
}

std::map<std::string, std::map<std::string, int, CompareNoUnderline>> Option::enum_map_;
std::map<std::string, std::map<int, std::string>> Option::enum_map_reverse_;

//初始化map，注意一些设置是有别名的
//所有字符小写！！！！
//第一项通常是默认项
void Option::initEnums()
{
    if (enum_map_.size() > 0)
    {
        return;
    }

    registerEnum<ActiveFunctionType>(
        {
            { "none", ACTIVE_FUNCTION_NONE },
            { "sigmoid", ACTIVE_FUNCTION_SIGMOID },
            { "relu", ACTIVE_FUNCTION_RELU },
            { "tanh", ACTIVE_FUNCTION_TANH },
            { "clipped_relu", ACTIVE_FUNCTION_CLIPPED_RELU },
            { "elu", ACTIVE_FUNCTION_ELU },
            //{ "scale", ACTIVE_FUNCTION_SCALE },
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
            { "constant", RANDOM_FILL_CONSTANT },
            { "xavier", RANDOM_FILL_XAVIER },
            { "gaussian", RANDOM_FILL_GAUSSIAN },
            { "msra", RANDOM_FILL_GAUSSIAN },
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

}    // namespace woco