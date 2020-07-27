#include "NetCifa.h"
#include "Log.h"
#include "Option.h"

namespace woco
{

NetCifa::NetCifa()
{
    registerFunctions();
}

cifa::Object NetCifa::registerMatrix(Matrix m)
{
    static int count;
    cifa::Object o(count++, "Matrix");
    map_matrix_[o] = std::move(m);
    return o;
}

cifa::Object NetCifa::registerLoss(MatrixOperator::Queue loss)
{
    static int count;
    cifa::Object o(count++, "Loss");
    map_loss_[o] = std::move(loss);
    return o;
}

std::vector<int> NetCifa::getVector(cifa::ObjectVector& v, int index)
{
    std::vector<int> r;
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
            r.push_back(o.value);
        }
    };
    expand(v[index]);
    return r;
}

void NetCifa::structure()
{
    //cudnnTensorDescriptor_t t;
    //auto s = sizeof(*t);
    weights_.clear();
    loss_.clear();
    runScript(message_);
    map_matrix_.clear();
    map_loss_.clear();

    for (auto& m : weights_)
    {
        MatrixExtend::fill(m, RANDOM_FILL_XAVIER, m.getChannel(), m.getNumber());
        //m.scale(0.3);
    }
    //addLoss(5e-4 * L2(weights_));
}

int NetCifa::runScript(const std::string& script)
{
    auto lines = convert::splitString(script, "\n");
    int i = 1;
    for (auto& l : lines)
    {
        Log::LOG("%3d\t%s\n", i++, l.c_str());
    }
    Log::LOG("\n");

    auto o = cifa_.run_script(script);
    return o.value;
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
            return registerMatrix(Matrix(dim));
        });
    cifa_.register_function("M", [this](cifa::ObjectVector& v)
        {
            std::vector<int> dim;
            for (int i = 0; i < v.size(); i++)
            {
                dim.push_back(v[i].value);
            }
            Matrix m(dim);
            addWeight(m);
            return registerMatrix(m);
        });
    cifa_.register_function("print_message", [this](cifa::ObjectVector& v)
        {
            fprintf(stdout, "%g\n", v[0].value);
            map_matrix_[v[0]].message();
            return cifa::Object();
        });
    cifa_.register_function("setXYA", [this](cifa::ObjectVector& v)
        {
            setXYA(map_matrix_[v[0]], map_matrix_[v[1]], map_matrix_[v[2]]);
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
                weights_.push_back(map_matrix_[o]);
            }
            return cifa::Object();
        });

    cifa_.user_add = [this](const cifa::Object& l, const cifa::Object& r)
    {
        if (l.type == "Matrix" && r.type == "Matrix")
        {
            return registerMatrix(map_matrix_[l] + map_matrix_[r]);
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
            return registerMatrix(map_matrix_[l] * map_matrix_[r]);
        }
        if (l.type == "Loss" || r.type == "Loss")
        {
            MatrixOperator::Queue q;
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
            auto stride = getVector(v, 2);
            auto padding = getVector(v, 3);
            return registerMatrix(conv(map_matrix_[v[0]], map_matrix_[v[1]], stride, padding));
        });
    cifa_.register_function("maxpool", [this](cifa::ObjectVector& v)
        {
            auto window = getVector(v, 1);
            auto stride = getVector(v, 2);
            auto padding = getVector(v, 3);
            return registerMatrix(maxpool(map_matrix_[v[0]], window, stride, padding));
        });
    cifa_.register_function("getRow", [this](cifa::ObjectVector& v)
        { return cifa::Object(map_matrix_[v[0]].getRow()); });
    cifa_.register_function("getChannel", [this](cifa::ObjectVector& v)
        { return cifa::Object(map_matrix_[v[0]].getChannel()); });
    cifa_.register_function("reshape", [this](cifa::ObjectVector& v)
        {
            auto dim = getVector(v, 1);
            return registerMatrix(reshape(map_matrix_[v[0]], dim));
        });
    cifa_.register_function("active", [this](cifa::ObjectVector& v)
        { return registerMatrix(active(map_matrix_[v[0]], ActiveFunctionType(int(v[1])))); });

    cifa_.register_function("addLoss", [this](cifa::ObjectVector& v)
        {
            for (auto& o : v)
            {
                loss_ = loss_ + map_loss_[o];
            }
            return cifa::Object();
        });
    cifa_.register_function("crossEntropy", [this](cifa::ObjectVector& v)
        { return registerLoss(crossEntropy(map_matrix_[v[0]], map_matrix_[v[1]])); });
    cifa_.register_function("L2", [this](cifa::ObjectVector& v)
        { return registerLoss(L2(map_matrix_[v[0]])); });

    for (int i = -1; i < 30; i++)
    {
        auto str = Option::getInstance().getStringFromEnum(ActiveFunctionType(i));
        if (str == "")
        {
            continue;
        }
        cifa_.register_parameter("active_" + str, i);
        cifa_.register_function(str, [this, i](cifa::ObjectVector& v)
            { return registerMatrix(active(map_matrix_[v[0]], ActiveFunctionType(i))); });
    }

    return 0;
}

}    // namespace woco