#include "NetCifa.h"
#include "Log.h"
#include "Option.h"

namespace cccc
{

NetCifa::NetCifa()
{
}

NetCifa::~NetCifa()
{
}

cifa::Object NetCifa::registerMatrix(MatrixSP&& m)
{
    cifa::Object o(count_++, "Matrix");
    map_matrix_[o] = std::move(m);
    return o;
}

cifa::Object NetCifa::registerLoss(std::vector<MatrixOp> loss)
{
    cifa::Object o(count_++, "Loss");
    map_loss_[o] = std::move(loss);
    return o;
}

std::vector<cifa::Object> NetCifa::getVector(cifa::ObjectVector& v, int index)
{
    std::vector<cifa::Object> r;
    if (index < 0 || index >= v.size())
    {
        return r;
    }
    std::function<void(cifa::Object&)> expand = [&r, &expand](cifa::Object& o)
    {
        if (o.v.size() > 0)
        {
            for (auto& o1 : o.v)
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
        r.push_back(o.value);
    }
    return r;
}

std::vector<real> NetCifa::getRealVector(cifa::ObjectVector& v, int index)
{
    auto vo = getVector(v, index);
    std::vector<real> r;
    for (auto& o : vo)
    {
        r.push_back(o.value);
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
    map_loss_.clear();

    runScript(option_->getString("cifa", "parameter"));
    if (runScript(option_->getString("cifa", "structure")))
    {
        LOG("Error in script!\n");
        return -1;
    }
    MatrixOp::simpleQueue(op_queue_, getX(), getA());
    if (op_queue_.size() == 0)
    {
        LOG("Empty compute queue!\n");
        return -2;
    }
    map_matrix_.clear();
#ifdef _DEBUG
    MatrixOp::print(op_queue_);
    MatrixOp::print(loss_);
#endif
    if (X_ && A_)
    {
        Y_->resize(*A_);
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
        for (auto key : option_->getAllKeys(section))
        {
            cifa_.register_parameter(strfunc::toLowerCase(section) + "::" + strfunc::toLowerCase(key), cifa::Object(option_->getReal(section, key), option_->getString(section, key)));
        }
    }

    auto lines = strfunc::splitString(strfunc::replaceAllSubString(script, "\r", ""), "\n");
    int i = 1;
    LOG::setLevel(option_->getInt("train", "output_net", 1));
    for (auto& l : lines)
    {
        LOG("{:3}\t{}\n", i++, l);
    }
    LOG("\n");
    auto o = cifa_.run_script(script);
    LOG::restoreLevel();
    if (o.type == "Error")
    {
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
                dim.push_back(v[i].value);
            }
            return registerMatrix(makeMatrixSP(dim));
        });
    cifa_.register_function("M", [this](cifa::ObjectVector& v)
        {
            std::vector<int> dim;
            for (int i = 0; i < v.size(); i++)
            {
                dim.push_back(v[i].value);
            }
            auto o = registerMatrix(makeMatrixSP(dim));
            return o;
        });
    cifa_.register_function("setValue", [this](cifa::ObjectVector& v)
        {
            auto& m = map_matrix_[v[0]];
            auto values = getVector(v, 1);
            std::vector<real> values_real(values.size());
            for (int i = 0; i < values.size(); i++)
            {
                values_real[i] = values[i].value;
            }
            m->importData(values_real.data(), values_real.size());
            return cifa::Object();
        });
    cifa_.register_function("repeat", [this](cifa::ObjectVector& v)
        {
            auto& m = map_matrix_[v[0]];
            m->repeat(1);
            return cifa::Object();
        });
    cifa_.register_function("scale", [this](cifa::ObjectVector& v)
        {
            auto& m = map_matrix_[v[0]];
            m->scale(v[1].value);
            return cifa::Object();
        });
    cifa_.register_function("print_message", [this](cifa::ObjectVector& v)
        {
            LOG("{}\n", v[0].value);
            map_matrix_[v[0]]->message();
            return cifa::Object();
        });
    cifa_.register_function("setXY", [this](cifa::ObjectVector& v)
        {
            //为方便理解，注意名字的区别
            setXA(map_matrix_[v[0]], map_matrix_[v[1]]);
            return cifa::Object();
        });
    cifa_.register_function("setLossWeight", [this](cifa::ObjectVector& v)
        {
            loss_weight_ = *map_matrix_[v[0]];
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
                weights_.push_back(map_matrix_[o].get());
            }
            return cifa::Object();
        });

    cifa_.user_add = [this](const cifa::Object& l, const cifa::Object& r)
    {
        if (l.type == "Matrix" && r.type == "Matrix")
        {
            MatrixOp op;
            auto o = registerMatrix(makeMatrixSP());
            if (map_matrix_[r]->getNumber() == 1 && map_matrix_[r]->getDataSize() != map_matrix_[l]->getDataSize())
            {
                op.as_addBias(map_matrix_[l], map_matrix_[r], map_matrix_[o]);
            }
            else
            {
                op.as_add(map_matrix_[l], map_matrix_[r], map_matrix_[o]);
            }
            op_queue_.push_back(op);
            return o;
        }
        if (l.type == "Loss" && r.type == "Loss")
        {
            return registerLoss(map_loss_[l] + map_loss_[r]);
        }
        return cifa::Object(l.value + r.value);
    };
    cifa_.user_mul = [this](const cifa::Object& l, const cifa::Object& r)
    {
        if (l.type == "Matrix" && r.type == "Matrix")
        {
            MatrixOp op;
            auto o = registerMatrix(makeMatrixSP());
            op.as_mul(map_matrix_[l], map_matrix_[r], map_matrix_[o]);
            op_queue_.push_back(op);
            return o;
        }
        if (l.type == "Loss" || r.type == "Loss")
        {
            std::vector<MatrixOp> q;
            if (l.type == "Loss")
            {
                q = r.value * map_loss_[l];
            }
            else
            {
                q = l.value * map_loss_[r];
            }
            return registerLoss(q);
        }
        return cifa::Object(l.value * r.value);
    };
    cifa_.register_function("conv", [this](cifa::ObjectVector& v)
        {
            auto stride = getIntVector(v, 2);
            auto padding = getIntVector(v, 3);
            MatrixOp op;
            auto o = registerMatrix(makeMatrixSP());
            op.as_conv(map_matrix_[v[0]], map_matrix_[v[1]], map_matrix_[o], stride, padding, 0);
            op_queue_.push_back(op);
            return o;
        });
    cifa_.register_function("corr", [this](cifa::ObjectVector& v)
        {
            auto stride = getIntVector(v, 2);
            auto padding = getIntVector(v, 3);
            MatrixOp op;
            auto o = registerMatrix(makeMatrixSP());
            op.as_conv(map_matrix_[v[0]], map_matrix_[v[1]], map_matrix_[o], stride, padding, 1);
            op_queue_.push_back(op);
            return o;
        });
    cifa_.register_function("maxpool", [this](cifa::ObjectVector& v)
        {
            auto window = getIntVector(v, 1);
            auto stride = getIntVector(v, 2);
            auto padding = getIntVector(v, 3);
            MatrixOp op;
            auto o = registerMatrix(makeMatrixSP());
            op.as_pool(map_matrix_[v[0]], map_matrix_[o], POOLING_MAX, 0, window, stride, padding);
            op_queue_.push_back(op);
            return o;
        });
    cifa_.register_function("reversepool", [this](cifa::ObjectVector& v)
        {
            auto window = getIntVector(v, 1);
            auto stride = getIntVector(v, 2);
            auto padding = getIntVector(v, 3);
            MatrixOp op;
            auto o = registerMatrix(makeMatrixSP());
            op.as_pool(map_matrix_[v[0]], map_matrix_[o], POOLING_AVERAGE_NOPADDING, 1, window, stride, padding);
            op_queue_.push_back(op);
            return o;
        });
    cifa_.register_function("width", [this](cifa::ObjectVector& v)
        { return cifa::Object(map_matrix_[v[0]]->getWidth()); });
    cifa_.register_function("height", [this](cifa::ObjectVector& v)
        { return cifa::Object(map_matrix_[v[0]]->getHeight()); });
    cifa_.register_function("row", [this](cifa::ObjectVector& v)
        { return cifa::Object(map_matrix_[v[0]]->getRow()); });
    cifa_.register_function("channel", [this](cifa::ObjectVector& v)
        { return cifa::Object(map_matrix_[v[0]]->getChannel()); });
    cifa_.register_function("reshape", [this](cifa::ObjectVector& v)
        {
            auto dim = getIntVector(v, 1);
            MatrixOp op;
            auto o = registerMatrix(makeMatrixSP());
            op.as_reshape(map_matrix_[v[0]], map_matrix_[o], dim);
            op_queue_.push_back(op);
            return o;
        });
    cifa_.register_function("mul", [this](cifa::ObjectVector& v)
        {
            auto dim = getIntVector(v, 2);
            MatrixOp op;
            auto o = registerMatrix(makeMatrixSP());
            op.as_mul(map_matrix_[v[0]], map_matrix_[v[1]], map_matrix_[o], 1, dim);
            op_queue_.push_back(op);
            return o;
        });
    cifa_.register_function("active", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            auto o = registerMatrix(makeMatrixSP());
            op.as_active(map_matrix_[v[0]], map_matrix_[o], ActiveFunctionType(int(v[1])), getIntVector(v, 2), getRealVector(v, 3), {});
            op_queue_.push_back(op);
            return o;
        });
    cifa_.register_function("concat", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            auto o = registerMatrix(makeMatrixSP());
            auto vo = getVector(v, 0);
            std::vector<MatrixSP> vm;
            for (auto& o1 : vo)
            {
                vm.push_back(map_matrix_[o1]);
            }
            op.as_concat(vm, map_matrix_[o]);
            op_queue_.push_back(op);
            return o;
        });

    cifa_.register_function("addLoss", [this](cifa::ObjectVector& v)
        {
            for (auto& o : v)
            {
                loss_ = loss_ + map_loss_[o];
            }
            return cifa::Object();
        });
    //cifa_.register_function("crossEntropy", [this](cifa::ObjectVector& v)
    //    { return registerLoss(crossEntropy(map_matrix_[v[0]], map_matrix_[v[1]])); });
    cifa_.register_function("L2", [this](cifa::ObjectVector& v)
        { return registerLoss(L2(map_matrix_[v[0]])); });

    for (int i = -1; i < 100; i++)
    {
        auto str = option_->getStringFromEnum(ActiveFunctionType(i));
        if (str == "")
        {
            continue;
        }
        cifa_.register_parameter("active_" + str, i);
        cifa_.register_function(str, [this, i](cifa::ObjectVector& v)
            {
                MatrixOp op;
                auto o = registerMatrix(makeMatrixSP());
                auto& Y = map_matrix_[o];
                op.as_active(map_matrix_[v[0]], Y, ActiveFunctionType(i), getIntVector(v, 1), getRealVector(v, 2), {});
                op_queue_.push_back(op);
                return o;
            });
    }

    return 0;
}

}    // namespace cccc