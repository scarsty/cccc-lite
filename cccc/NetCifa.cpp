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
    if (index < 0 || index >= v.size() || !v[index].hasValue())
    {
        return r;
    }
    std::function<void(const cifa::Object&)> expand = [&r, &expand](const cifa::Object& o)
    {
        if (o.isType<std::vector<cifa::Object>>())
        {
            for (auto& o1 : o.ref<std::vector<cifa::Object>>())
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

    auto lines = strfunc::splitString(strfunc::replaceAllSubString(script, "\r", ""), "\n", false);
    int i = 1;
    if (option_->getInt("train", "output_net", 1))
    {
        for (auto& l : lines)
        {
            LOG("{:3}\t{}\n", i++, l);
        }
        LOG("\n");
    }
    cifa_.set_output_error(true);
    auto o = cifa_.run_script(script, cifa_parameters_);
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
    registerFunction("Matrix", [this](cifa::ObjectVector& v)
        {
            std::vector<int> dim;
            for (int i = 0; i < v.size(); i++)
            {
                dim.push_back(v[i]);
            }
            return cifa::Object(makeMatrixSPWithState(dim));
        });
    registerFunction("M", [this](cifa::ObjectVector& v)
        {
            std::vector<int> dim;
            for (int i = 0; i < v.size(); i++)
            {
                dim.push_back(v[i]);
            }
            return cifa::Object(makeMatrixSPWithState(dim));
        });
    //registerFunction("Mb", [this](cifa::ObjectVector& v)
    //    {
    //        std::vector<int> dim = getIntVector(v, 0);
    //        auto o = cifa::Object(makeMatrixSPWithState(dim));
    //        o.to<MatrixSP>()->setNeedBack(v[1].toInt());
    //        o.to<MatrixSP>()->setNeedLoad(need_load_state_);
    //        return o;
    //    });
    registerFunction("setValue", [this](cifa::ObjectVector& v)
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
    registerFunction("repeat", [this](cifa::ObjectVector& v)
        {
            auto m = v[0].to<MatrixSP>();
            m->repeat(1);
            return cifa::Object();
        });
    registerFunction("scale", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_scale(v[0].to<MatrixSP>(), o.to<MatrixSP>(), float(v[1].toDouble()));
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("print_message", [this](cifa::ObjectVector& v)
        {
            LOG("{}\n", v[0].toDouble());
            v[0].to<MatrixSP>()->message();
            return cifa::Object();
        });
    registerFunction("setXY", [this](cifa::ObjectVector& v)
        {
            //为方便理解，注意名字的区别
            setXA(v[0].to<MatrixSP>(), v[1].to<MatrixSP>());
            return cifa::Object();
        });
    registerFunction("setXA", [this](cifa::ObjectVector& v)
        {
            setXA(v[0].to<MatrixSP>(), v[1].to<MatrixSP>());
            return cifa::Object();
        });
    registerFunction("setLossWeight", [this](cifa::ObjectVector& v)
        {
            extra_matrixsp_["loss_weight"] = v[0].to<MatrixSP>();
            return cifa::Object();
        });
    registerFunction("clearWeight", [this](cifa::ObjectVector& v)
        {
            weights_.clear();
            return cifa::Object();
        });
    registerFunction("addWeight", [this](cifa::ObjectVector& v)
        {
            for (auto& o : v)
            {
                weights_.push_back(o.to<MatrixSP>().get());
            }
            return cifa::Object();
        });
    registerFunction("addBias", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_addBias(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), o.to<MatrixSP>());
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("add", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            if (v[0].isType<MatrixSP>())
            {
                op.as_add(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), o.to<MatrixSP>());
            }
            else
            {
                auto vo = getVector(v, 0);
                std::vector<MatrixSP> vm;
                for (auto& o1 : vo)
                {
                    vm.push_back(o1.to<MatrixSP>());
                }
                std::vector<float> va;
                if (v.size() >= 2)
                {
                    va = getRealVector(v, 1);
                }
                op.as_add(vm, o.to<MatrixSP>(), va);
            }
            op_queue_.push_back(op);
            return o;
        });
    cifa_.user_add.push_back([this](const cifa::Object& l, const cifa::Object& r)
        {
            if (l.isType<MatrixSP>() && r.isType<MatrixSP>())
            {
                MatrixOp op;
                op.solver_type_ = current_solver_type_;
                auto o = cifa::Object(makeMatrixSPWithState());
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
                op.solver_type_ = current_solver_type_;
                auto o = cifa::Object(makeMatrixSPWithState());
                op.as_mul(l.to<MatrixSP>(), r.to<MatrixSP>(), o.to<MatrixSP>());
                op_queue_.push_back(op);
                return o;
            }
            if (l.isType<MatrixSP>() && r.isNumber())
            {
                MatrixOp op;
                op.solver_type_ = current_solver_type_;
                auto o = cifa::Object(makeMatrixSPWithState());
                op.as_scale(l.to<MatrixSP>(), o.to<MatrixSP>(), float(r.toDouble()));
                op_queue_.push_back(op);
                return o;
            }
            if (r.isType<MatrixSP>() && l.isNumber())
            {
                MatrixOp op;
                op.solver_type_ = current_solver_type_;
                auto o = cifa::Object(makeMatrixSPWithState());
                op.as_scale(r.to<MatrixSP>(), o.to<MatrixSP>(), float(l.toDouble()));
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
    registerFunction("conv", [this](cifa::ObjectVector& v)
        {
            auto stride = getIntVector(v, 2);
            auto padding = getIntVector(v, 3);
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            auto conv_algo = option_->getInt("train", "conv_algo", -1);
            op.as_conv(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), o.to<MatrixSP>(), stride, padding, conv_algo);
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("corr", [this](cifa::ObjectVector& v)
        {
            auto stride = getIntVector(v, 2);
            auto padding = getIntVector(v, 3);
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_corr(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), o.to<MatrixSP>(), stride, padding, 1);
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("pool", [this](cifa::ObjectVector& v)
        {
            auto window = getIntVector(v, 2);
            auto stride = getIntVector(v, 3);
            auto padding = getIntVector(v, 4);
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_pool(v[0].to<MatrixSP>(), o.to<MatrixSP>(), PoolingType(int(v[1])), POOLING_NOT_REVERSE, window, stride, padding);
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("maxPool", [this](cifa::ObjectVector& v)
        {
            auto window = getIntVector(v, 1);
            auto stride = getIntVector(v, 2);
            auto padding = getIntVector(v, 3);
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_pool(v[0].to<MatrixSP>(), o.to<MatrixSP>(), POOLING_MAX, POOLING_NOT_REVERSE, window, stride, padding);
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("averagePool", [this](cifa::ObjectVector& v)
        {
            auto window = getIntVector(v, 1);
            auto stride = getIntVector(v, 2);
            auto padding = getIntVector(v, 3);
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_pool(v[0].to<MatrixSP>(), o.to<MatrixSP>(), POOLING_AVERAGE_NOPADDING, POOLING_NOT_REVERSE, window, stride, padding);
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("reversePool", [this](cifa::ObjectVector& v)
        {
            auto window = getIntVector(v, 1);
            auto stride = getIntVector(v, 2);
            auto padding = getIntVector(v, 3);
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_pool(v[0].to<MatrixSP>(), o.to<MatrixSP>(), POOLING_AVERAGE_NOPADDING, POOLING_REVERSE, window, stride, padding);
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("poolChannel", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_poolChannel(v[0].to<MatrixSP>(), o.to<MatrixSP>(), PoolingType(int(v[1])), POOLING_NOT_REVERSE);
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("width", [this](cifa::ObjectVector& v)
        {
            return cifa::Object(v[0].to<MatrixSP>()->getWidth());
        });
    registerFunction("height", [this](cifa::ObjectVector& v)
        {
            return cifa::Object(v[0].to<MatrixSP>()->getHeight());
        });
    registerFunction("channel", [this](cifa::ObjectVector& v)
        {
            return cifa::Object(v[0].to<MatrixSP>()->getChannel());
        });
    registerFunction("row", [this](cifa::ObjectVector& v)
        {
            return cifa::Object(v[0].to<MatrixSP>()->getRow());
        });
    registerFunction("number", [this](cifa::ObjectVector& v)
        {
            return cifa::Object(v[0].to<MatrixSP>()->getNumber());
        });
    registerFunction("reshape", [this](cifa::ObjectVector& v)
        {
            auto dim = getIntVector(v, 1);
            MatrixOp op;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_reshape(v[0].to<MatrixSP>(), o.to<MatrixSP>(), dim);
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("setNeedBack", [this](cifa::ObjectVector& v)
        {
            v[0].to<MatrixSP>()->setNeedBack(v[1].toInt());
            return v[1];
        });
    registerFunction("setAllNeedBack", [this](cifa::ObjectVector& v)
        {
            auto back = v[0].toInt();
            for (auto& op : op_queue_)
            {
                for (auto m : op.getMatrixIn())
                {
                    m->setNeedBack(back);
                }
                for (auto m : op.getMatrixOut())
                {
                    m->setNeedBack(back);
                }
            }
            return cifa::Object();
        });
    registerFunction("setNeedBackState", [this](cifa::ObjectVector& v)
        {
            need_back_state_ = v[0].toInt();
            return cifa::Object();
        });
    registerFunction("setNeedLoadState", [this](cifa::ObjectVector& v)
        {
            need_load_state_ = v[0].toInt();
            return cifa::Object();
        });
    registerFunction("setCurrentSolverType", [this](cifa::ObjectVector& v)
        {
            current_solver_type_ = option_->getEnumFromString<SolverType>(v[0].toString());
            return cifa::Object();
        });
    registerFunction("mul", [this](cifa::ObjectVector& v)
        {
            auto dim = getIntVector(v, 2);
            MatrixOp op;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_mul(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), o.to<MatrixSP>(), 1, dim);
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("elementMul", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            double a = 1.0;
            if (v.size() >= 3)
            {
                a = v[2].toDouble();
            }
            op.as_elementMul(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), o.to<MatrixSP>(), a);
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("active", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_active(v[0].to<MatrixSP>(), o.to<MatrixSP>(), ActiveFunctionType(int(v[1])), getIntVector(v, 2), getRealVector(v, 3), {});
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("concat", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            if (v[0].isType<MatrixGroup>())
            {
                auto vm = v[0].to<MatrixGroup>();
                op.as_concat(vm, o.to<MatrixSP>());
            }
            else
            {
                MatrixGroup vm;
                if (v[0].isType<std::string>())
                {
                    int i = 0;
                    while (true)
                    {
                        auto name = v[0].toString() + "[" + std::to_string(i) + "]";
                        if (cifa_parameters_.contains(name))
                        {
                            vm.push_back(cifa_parameters_[name].to<MatrixSP>());
                            i++;
                        }
                        else
                        {
                            break;
                        }
                    }
                }
                else
                {
                    auto vo = getVector(v, 0);
                    for (auto& o1 : vo)
                    {
                        vm.push_back(o1.to<MatrixSP>());
                    }
                }
                op.as_concat(vm, o.to<MatrixSP>());
            }
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("max", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_max(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), o.to<MatrixSP>());
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("addLoss", [this](cifa::ObjectVector& v)
        {
            for (auto& o : v)
            {
                loss_ = loss_ + o.to<Loss>();
            }
            return cifa::Object();
        });
    registerFunction("crossEntropy", [this](cifa::ObjectVector& v)
        {
            if (v.size() == 2)
            {
                return Loss(crossEntropy(v[0].to<MatrixSP>(), v[1].to<MatrixSP>()));
            }
            else if (v.size() == 3)
            {
                return Loss(crossEntropy(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), v[2].to<MatrixSP>()));
            }
        });
    registerFunction("focal", [this](cifa::ObjectVector& v)
        {
            if (v.size() == 2)
            {
                return Loss(focal(v[0].to<MatrixSP>(), v[1].to<MatrixSP>()));
            }
            else if (v.size() == 3)
            {
                return Loss(focal(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), v[2].to<MatrixSP>()));
            }
        });
    registerFunction("L2", [this](cifa::ObjectVector& v)
        {
            return Loss(L2(v[0].to<MatrixSP>()));
        });
    registerFunction("crossEntropy2", [this](cifa::ObjectVector& v)
        {
            std::vector<MatrixSP> vm;
            for (auto& o : getVector(v, 0))
            {
                vm.push_back(o.to<MatrixSP>());
            }
            return Loss(commonLoss(MatrixOpType::LOSS, vm, getRealVector(v, 1)));
        });
    registerFunction("commonloss", [this](cifa::ObjectVector& v)
        {
            std::vector<MatrixSP> vm;
            for (auto& o : getVector(v, 1))
            {
                vm.push_back(o.to<MatrixSP>());
            }
            return Loss(commonLoss(MatrixOpType(int(v[0])), vm, getRealVector(v, 2)));
        });
    registerFunction("matrixGroup", [this](cifa::ObjectVector& v)
        {
            return MatrixGroup({});
        });
    registerFunction("addIntoGroup", [this](cifa::ObjectVector& v)
        {
            v[0].to<MatrixGroup>().push_back(v[1].to<MatrixSP>());
            return cifa::Object();
        });
    registerFunction("getA", [this](cifa::ObjectVector& v)
        {
            return cifa::Object(A_);
        });
    registerFunction("getX", [this](cifa::ObjectVector& v)
        {
            return cifa::Object(X_);
        });
    registerFunction("getY", [this](cifa::ObjectVector& v)
        {
            return cifa::Object(Y_);
        });
    registerFunction("getLossWeight", [this](cifa::ObjectVector& v)
        {
            return cifa::Object(extra_matrixsp_["loss_weight"]);
        });
    registerFunction("getMatrix", [this](cifa::ObjectVector& v)
        {
            return cifa::Object(getMatrixByName(v[0].toString()));
        });
    registerFunction("haveMatrix", [this](cifa::ObjectVector& v)
        {
            return cifa::Object(extra_matrixsp_.contains(v[0].toString()));
        });
    registerFunction("showMatrix", [this](cifa::ObjectVector& v)
        {
            extra_matrixsp_[v[0].toString()] = v[1].to<MatrixSP>();
            return cifa::Object();
        });
    for (int i = -1; i < 100; i++)
    {
        auto str = option_->getStringFromEnum(ActiveFunctionType(i));
        if (str != "")
        {
            cifa_.register_parameter("active_" + str, i);
            registerFunction(str, [this, i](cifa::ObjectVector& v)
                {
                    MatrixOp op;
                    auto o = cifa::Object(makeMatrixSPWithState());
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
    for (int i = -1; i < 100; i++)
    {
        auto str = MatrixOp::getOpName(MatrixOpType(i));
        if (str != "")
        {
            cifa_.register_parameter("op_" + str, i);
        }
    }
    registerFunction("setOption", [this](cifa::ObjectVector& v)
        {
            option_->setKey(v[0].toString(), v[1].toString(), v[2].toString());
            return cifa::Object();
        });

    return 0;
}

inline void NetCifa::registerFunction(std::string name, cifa::Cifa::func_type func)
{
    cifa_.register_function(name, func);
    auto str_lower = strfunc::toLowerCase(name);
    if (str_lower != name)
    {
        cifa_.register_function(str_lower, func);    //再注册一个小写版本，兼容性需求
    }
}

void NetCifa::setXA(const MatrixSP& X, const MatrixSP& A)
{
    X_ = X;
    A_ = A;
}
}    // namespace cccc