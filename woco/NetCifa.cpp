#include "NetCifa.h"

namespace woco
{

NetCifa::NetCifa()
{
    registerFunctions();
}

cifa::Object NetCifa::registerMatrix(Matrix& m)
{
    static int count;
    cifa::Object o(count++, "Matrix");
    map_matrix_[o] = std::move(m);
    return o;
}

cifa::Object NetCifa::registerLoss(MatrixOperator::Queue& loss)
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
    for (auto& o : v[index].v)
    {
        r.push_back(o.value);
    }
    return r;
}

void NetCifa::structure()
{
    //cudnnTensorDescriptor_t t;
    //auto s = sizeof(*t);
    weights_.clear();
    loss_.clear();
    runScript(script_);
    map_matrix_.clear();
    map_loss_.clear();

    for (auto& m : weights_)
    {
        MatrixExtend::fill(m, RANDOM_FILL_XAVIER, m.getChannel(), m.getNumber());
    }
}

int NetCifa::runScript(const std::string& script)
{
    auto o = cifa_.run_script(script_);
    return o.value;
}

int NetCifa::registerFunctions()
{
    cifa_.register_user_data("__this", this);
    cifa_.register_function("Matrix", [&](cifa::ObjectVector& v) -> cifa::Object
        {
            std::vector<int> dim;
            for (int i = 0; i < v.size(); i++)
            {
                dim.push_back(v[i].value);
            }
            Matrix m(dim);
            return registerMatrix(m);
        });
    cifa_.register_function("print_message", [&](cifa::ObjectVector& v) -> cifa::Object
        {
            fprintf(stdout, "%g\n", v[0].value);
            map_matrix_[v[0]].message();
            return cifa::Object();
        });
    cifa_.register_function("setXYA", [&](cifa::ObjectVector& v) -> cifa::Object
        {
            setXYA(map_matrix_[v[0]], map_matrix_[v[1]], map_matrix_[v[2]]);
            return cifa::Object();
        });
    cifa_.register_function("clearWeight", [&](cifa::ObjectVector& v) -> cifa::Object
        {
            weights_.clear();
            return cifa::Object();
        });
    cifa_.register_function("addWeight", [&](cifa::ObjectVector& v) -> cifa::Object
        {
            for (auto& o : v)
            {
                weights_.push_back(map_matrix_[o]);
            }
            return cifa::Object();
        });

    cifa_.user_add = [&](const cifa::Object& l, const cifa::Object& r) -> cifa::Object
    {
        if (l.type == "Matrix" && r.type == "Matrix")
        {
            Matrix m = map_matrix_[l] + map_matrix_[r];
            return registerMatrix(m);
        }
        if (l.type == "Loss" && r.type == "Loss")
        {
            MatrixOperator::Queue q = map_loss_[l] + map_loss_[r];
            return registerLoss(q);
        }
        return cifa::Object(l.value + r.value);
    };
    cifa_.user_mul = [&](const cifa::Object& l, const cifa::Object& r) -> cifa::Object
    {
        if (l.type == "Matrix" && r.type == "Matrix")
        {
            Matrix m = map_matrix_[l] * map_matrix_[r];
            return registerMatrix(m);
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
    cifa_.register_function("conv", [&](cifa::ObjectVector& v) -> cifa::Object
        {
            auto stride = getVector(v, 2);
            auto padding = getVector(v, 3);
            Matrix m = conv(map_matrix_[v[0]], map_matrix_[v[1]], stride, padding);
            return registerMatrix(m);
        });
    cifa_.register_function("maxpool", [&](cifa::ObjectVector& v) -> cifa::Object
        {
            auto window = getVector(v, 1);
            auto stride = getVector(v, 2);
            auto padding = getVector(v, 3);
            Matrix m = maxpool(map_matrix_[v[0]], window, stride, padding);
            return registerMatrix(m);
        });
    cifa_.register_function("getRow", [&](cifa::ObjectVector& v) -> cifa::Object
        {
            cifa::Object o(map_matrix_[v[0]].getRow());
            return o;
        });

#define REGISTER(func) \
    cifa_.register_function(#func, [&](cifa::ObjectVector& v) -> cifa::Object { \
        Matrix m = func(map_matrix_[v[0]]); \
        return registerMatrix(m); \
    })

    REGISTER(relu);
    REGISTER(sigmoid);
    REGISTER(softmax);
    REGISTER(softmax_ce);

    cifa_.register_function("addLoss", [&](cifa::ObjectVector& v) -> cifa::Object
        {
            for (auto& o : v)
            {
                loss_ = loss_ + map_loss_[o];
            }
            return cifa::Object();
        });
    cifa_.register_function("crossEntropy", [&](cifa::ObjectVector& v) -> cifa::Object
        {
            auto q = crossEntropy(map_matrix_[v[0]], map_matrix_[v[1]]);
            return registerLoss(q);
        });
    cifa_.register_function("L2", [&](cifa::ObjectVector& v) -> cifa::Object
        {
            auto q = L2(map_matrix_[v[0]]);
            return registerLoss(q);
        });

    return 0;
}

}    // namespace woco