#include "NetCifa.h"
#include "Log.h"
#include "Option.h"
#include "strfunc.h"

namespace cccc
{

NetCifa::NetCifa()
{
}

NetCifa::~NetCifa()
{
}

std::vector<cifa::Object> NetCifa::getVector(cifa::ObjectVector& v, int index)
{
    std::vector<cifa::Object> r;
    //可能是一个union中的逗号表达式
    //cifa的计算，如果union中只有一个，则直接返回，如果是逗号表达式，则返回一个递归形式的结果
    if (index < 0 || index >= v.size() || (!v[index].hasValue() && v[index].subV().empty()))
    {
        return r;
    }
    std::function<void(const cifa::Object&)> expand = [&r, &expand](const cifa::Object& o)
    {
        if (o.subV().size() > 0)
        {
            for (auto& o1 : o.subV())
            {
                expand(o1);
            }
        }
        else
        {
            r.push_back(o);
        }
    };
    expand(v[index]);
    return r;
}

std::vector<int> NetCifa::getIntVector(cifa::ObjectVector& v, int index)
{
    auto vo = getVector(v, index);
    std::vector<int> r;
    for (auto& o : vo)
    {
        r.push_back(o.toInt());
    }
    return r;
}

std::vector<float> NetCifa::getRealVector(cifa::ObjectVector& v, int index)
{
    auto vo = getVector(v, index);
    std::vector<float> r;
    for (auto& o : vo)
    {
        r.push_back(o.toDouble());
    }
    return r;
}

int NetCifa::init2()
{
    setDeviceSelf();
    registerFunctions();

    //cudnnTensorDescriptor_t t;
    //auto s = sizeof(*t);
    loss_.clear();
    //map_loss_.clear();

    runScript(option_->getString("cifa", "parameter"));
    if (runScript(option_->getString("cifa", "structure")))
    {
        LOG("Error in script!\n");
        return -1;
    }
    //map_matrix_.clear();
#ifdef _DEBUG
    //LOG("{}\n", MatrixOp::ir(op_queue_));
    //MatrixOp::ir(loss_);
#endif
    if (X_ && A_)
    {
        Y_->resize(A_->getDim());
    }
    else
    {
        LOG("Please set X, Y!\n");
        return -4;
    }
    return 0;
}

int NetCifa::runScript(const std::string& script)
{
    for (auto section : option_->getAllSections())
    {
        std::map<std::string, cifa::Object> m;
        for (auto key : option_->getAllKeys(section))
        {
            m[strfunc::toLowerCase(key)] = cifa::Object(option_->INIReaderNoUnderline::getReal(section, key), option_->INIReaderNoUnderline::getString(section, key));
        }
        cifa_.register_parameter(strfunc::toLowerCase(section), m);
    }

    auto lines = strfunc::splitString(strfunc::replaceAllSubString(script, "\r", ""), "\n");
    int i = 1;
    if (option_->getInt("train", "output_net", 1))
    {
        for (auto& l : lines)
        {
            LOG("{:3}\t{}\n", i++, l);
        }
        LOG("\n");
    }
    cifa_.set_output_error(false);
    auto o = cifa_.run_script(script);
    if (o.getSpecialType() == "Error" && o.isType<std::string>() && o.toString().find("Error") == 0)
    {
        LOG("{}", o.toString());
        return -1;
    }
    return 0;
}

int NetCifa::registerFunctions()
{
    cifa_.register_user_data("__this", this);
    cifa_.register_function("Matrix", [this](cifa::ObjectVector& v)
        {
            std::vector<int> dim;
            for (int i = 0; i < v.size(); i++)
            {
                dim.push_back(v[i]);
            }
            return cifa::Object(makeMatrixSP(dim));
        });
    cifa_.register_function("M", [this](cifa::ObjectVector& v)
        {
            std::vector<int> dim;
            for (int i = 0; i < v.size(); i++)
            {
                dim.push_back(v[i]);
            }
            return cifa::Object(makeMatrixSP(dim));
        });
    cifa_.register_function("Mb", [this](cifa::ObjectVector& v)
        {
            std::vector<int> dim = getIntVector(v, 0);
            auto o = cifa::Object(makeMatrixSP(dim));
            o.to<MatrixSP>()->setNeedBack(v[1].toInt());
            return o;
        });
    cifa_.register_function("setValue", [this](cifa::ObjectVector& v)
        {
            auto m = v[0].to<MatrixSP>();
            auto values = getVector(v, 1);
            std::vector<float> values_real(values.size());
            for (int i = 0; i < values.size(); i++)
            {
                values_real[i] = values[i].toDouble();
            }
            m->importData(values_real.data(), values_real.size());
            return cifa::Object();
        });
    cifa_.register_function("repeat", [this](cifa::ObjectVector& v)
        {
            auto m = v[0].to<MatrixSP>();
            m->repeat(1);
            return cifa::Object();
        });
    cifa_.register_function("scale", [this](cifa::ObjectVector& v)
        {
            auto m = v[0].to<MatrixSP>();
            m->scale(v[1].toDouble());
            return cifa::Object();
        });
    cifa_.register_function("print_message", [this](cifa::ObjectVector& v)
        {
            LOG("{}\n", v[0].toDouble());
            v[0].to<MatrixSP>()->message();
            return cifa::Object();
        });
    cifa_.register_function("setXY", [this](cifa::ObjectVector& v)
        {
            //为方便理解，注意名字的区别
            setXA(v[0].to<MatrixSP>(), v[1].to<MatrixSP>());
            return cifa::Object();
        });
    cifa_.register_function("setLossWeight", [this](cifa::ObjectVector& v)
        {
            loss_weight_ = v[0].to<MatrixSP>();
            return cifa::Object();
        });
    cifa_.register_function("clearWeight", [this](cifa::ObjectVector& v)
        {
            weights_.clear();
            return cifa::Object();
        });
    cifa_.register_function("addWeight", [this](cifa::ObjectVector& v)
        {
            for (auto& o : v)
            {
                weights_.push_back(o.to<MatrixSP>().get());
            }
            return cifa::Object();
        });
    cifa_.register_function("addBias", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            auto o = cifa::Object(makeMatrixSP());
            op.as_addBias(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), o.to<MatrixSP>());
            op_queue_.push_back(op);
            return o;
        });
    cifa_.register_function("add", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            auto o = cifa::Object(makeMatrixSP());
            op.as_add(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), o.to<MatrixSP>());
            op_queue_.push_back(op);
            return o;
        });
    cifa_.user_add.push_back([this](const cifa::Object& l, const cifa::Object& r)
        {
            if (l.isType<MatrixSP>() && r.isType<MatrixSP>())
            {
                MatrixOp op;
                auto o = cifa::Object(makeMatrixSP());
                if (r.to<MatrixSP>()->getNumber() == 1 && r.to<MatrixSP>()->getDataSize() != l.to<MatrixSP>()->getDataSize())    //注意这个判断不严格
                {
                    op.as_addBias(l.to<MatrixSP>(), r.to<MatrixSP>(), o.to<MatrixSP>());
                }
                else
                {
                    op.as_add(l.to<MatrixSP>(), r.to<MatrixSP>(), o.to<MatrixSP>());
                }
                op_queue_.push_back(op);
                return o;
            }
            if (l.isType<Loss>() && r.isType<Loss>())
            {
                return cifa::Object(l.to<Loss>() + r.to<Loss>());
            }
            return cifa::Object();
        });
    cifa_.user_mul.push_back([this](const cifa::Object& l, const cifa::Object& r)
        {
            if (l.isType<MatrixSP>() && r.isType<MatrixSP>())
            {
                MatrixOp op;
                auto o = cifa::Object(makeMatrixSP());
                op.as_mul(l.to<MatrixSP>(), r.to<MatrixSP>(), o.to<MatrixSP>());
                op_queue_.push_back(op);
                return o;
            }
            if (l.isType<Loss>() || r.isType<Loss>())
            {
                Loss q;
                if (l.isType<Loss>())
                {
                    q = r.toDouble() * l.to<Loss>();
                }
                else
                {
                    q = l.toDouble() * r.to<Loss>();
                }
                return cifa::Object(q);
            }
            return cifa::Object();
        });
    cifa_.register_function("conv", [this](cifa::ObjectVector& v)
        {
            auto stride = getIntVector(v, 2);
            auto padding = getIntVector(v, 3);
            MatrixOp op;
            auto o = cifa::Object(makeMatrixSP());
            auto conv_algo = option_->getInt("train", "conv_algo", -1);
            op.as_conv(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), o.to<MatrixSP>(), stride, padding, conv_algo);
            op_queue_.push_back(op);
            return o;
        });
    cifa_.register_function("corr", [this](cifa::ObjectVector& v)
        {
            auto stride = getIntVector(v, 2);
            auto padding = getIntVector(v, 3);
            MatrixOp op;
            auto o = cifa::Object(makeMatrixSP());
            op.as_conv(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), o.to<MatrixSP>(), stride, padding, 1);
            op_queue_.push_back(op);
            return o;
        });
    cifa_.register_function("pool", [this](cifa::ObjectVector& v)
        {
            auto window = getIntVector(v, 2);
            auto stride = getIntVector(v, 3);
            auto padding = getIntVector(v, 4);
            MatrixOp op;
            auto o = cifa::Object(makeMatrixSP());
            op.as_pool(v[0].to<MatrixSP>(), o.to<MatrixSP>(), PoolingType(int(v[1])), POOLING_NOT_REVERSE, window, stride, padding);
            op_queue_.push_back(op);
            return o;
        });
    cifa_.register_function("maxpool", [this](cifa::ObjectVector& v)
        {
            auto window = getIntVector(v, 1);
            auto stride = getIntVector(v, 2);
            auto padding = getIntVector(v, 3);
            MatrixOp op;
            auto o = cifa::Object(makeMatrixSP());
            op.as_pool(v[0].to<MatrixSP>(), o.to<MatrixSP>(), POOLING_MAX, POOLING_NOT_REVERSE, window, stride, padding);
            op_queue_.push_back(op);
            return o;
        });
    cifa_.register_function("averagepool", [this](cifa::ObjectVector& v)
        {
            auto window = getIntVector(v, 1);
            auto stride = getIntVector(v, 2);
            auto padding = getIntVector(v, 3);
            MatrixOp op;
            auto o = cifa::Object(makeMatrixSP());
            op.as_pool(v[0].to<MatrixSP>(), o.to<MatrixSP>(), POOLING_AVERAGE_NOPADDING, POOLING_NOT_REVERSE, window, stride, padding);
            op_queue_.push_back(op);
            return o;
        });
    cifa_.register_function("reversepool", [this](cifa::ObjectVector& v)
        {
            auto window = getIntVector(v, 1);
            auto stride = getIntVector(v, 2);
            auto padding = getIntVector(v, 3);
            MatrixOp op;
            auto o = cifa::Object(makeMatrixSP());
            op.as_pool(v[0].to<MatrixSP>(), o.to<MatrixSP>(), POOLING_AVERAGE_NOPADDING, POOLING_REVERSE, window, stride, padding);
            op_queue_.push_back(op);
            return o;
        });
    cifa_.register_function("width", [this](cifa::ObjectVector& v)
        {
            return cifa::Object(v[0].to<MatrixSP>()->getWidth());
        });
    cifa_.register_function("height", [this](cifa::ObjectVector& v)
        {
            return cifa::Object(v[0].to<MatrixSP>()->getHeight());
        });
    cifa_.register_function("row", [this](cifa::ObjectVector& v)
        {
            return cifa::Object(v[0].to<MatrixSP>()->getRow());
        });
    cifa_.register_function("channel", [this](cifa::ObjectVector& v)
        {
            return cifa::Object(v[0].to<MatrixSP>()->getChannel());
        });
    cifa_.register_function("reshape", [this](cifa::ObjectVector& v)
        {
            auto dim = getIntVector(v, 1);
            MatrixOp op;
            auto o = cifa::Object(makeMatrixSP());
            op.as_reshape(v[0].to<MatrixSP>(), o.to<MatrixSP>(), dim);
            op_queue_.push_back(op);
            return o;
        });
    cifa_.register_function("setneedback", [this](cifa::ObjectVector& v)
        {
            v[0].to<MatrixSP>()->setNeedBack(v[1].toInt());
            return v[1];
        });
    cifa_.register_function("mul", [this](cifa::ObjectVector& v)
        {
            auto dim = getIntVector(v, 2);
            MatrixOp op;
            auto o = cifa::Object(makeMatrixSP());
            op.as_mul(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), o.to<MatrixSP>(), 1, dim);
            op_queue_.push_back(op);
            return o;
        });
    cifa_.register_function("active", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            auto o = cifa::Object(makeMatrixSP());
            op.as_active(v[0].to<MatrixSP>(), o.to<MatrixSP>(), ActiveFunctionType(int(v[1])), getIntVector(v, 2), getRealVector(v, 3), {});
            op_queue_.push_back(op);
            return o;
        });
    cifa_.register_function("concat", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            auto o = cifa::Object(makeMatrixSP());
            if (v.size() == 1 && v[0].isType<MatrixGroup>())
            {
                auto vm = v[0].to<MatrixGroup>();
                op.as_concat(vm, o.to<MatrixSP>());
            }
            else
            {
                auto vo = getVector(v, 0);
                std::vector<MatrixSP> vm;
                for (auto& o1 : vo)
                {
                    vm.push_back(o1.to<MatrixSP>());
                }
                op.as_concat(vm, o.to<MatrixSP>());
            }
            op_queue_.push_back(op);
            return o;
        });
    cifa_.register_function("max", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            auto o = cifa::Object(makeMatrixSP());
            op.as_max(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), o.to<MatrixSP>());
            op_queue_.push_back(op);
            return o;
        });
    cifa_.register_function("addLoss", [this](cifa::ObjectVector& v)
        {
            for (auto& o : v)
            {
                loss_ = loss_ + o.to<Loss>();
            }
            return cifa::Object();
        });
    //cifa_.register_function("crossEntropy", [this](cifa::ObjectVector& v)
    //    { return registerLoss(crossEntropy(v[0].to<MatrixSP>(), v[1].to<MatrixSP>())); });
    cifa_.register_function("L2", [this](cifa::ObjectVector& v)
        {
            return Loss(L2(v[0].to<MatrixSP>()));
        });
    cifa_.register_function("MatrixGroup", [this](cifa::ObjectVector& v)
        {
            return MatrixGroup({});
        });
    cifa_.register_function("addIntoGroup", [this](cifa::ObjectVector& v)
        {
            v[0].to<MatrixGroup>().push_back(v[1].to<MatrixSP>());
            return cifa::Object();
        });
    for (int i = -1; i < 100; i++)
    {
        auto str = option_->getStringFromEnum(ActiveFunctionType(i));
        if (str != "")
        {
            cifa_.register_parameter("active_" + str, i);
            cifa_.register_function(str, [this, i](cifa::ObjectVector& v)
                {
                    MatrixOp op;
                    auto o = cifa::Object(makeMatrixSP());
                    auto Y = o.to<MatrixSP>();
                    op.as_active(v[0].to<MatrixSP>(), Y, ActiveFunctionType(i), getIntVector(v, 1), getRealVector(v, 2), {});
                    op_queue_.push_back(op);
                    return o;
                });
        }
    }
    for (int i = -1; i < 100; i++)
    {
        auto str = option_->getStringFromEnum(PoolingType(i));
        if (str != "")
        {
            cifa_.register_parameter("pool_" + str, i);
        }
    }

    return 0;
}

}    // namespace cccc