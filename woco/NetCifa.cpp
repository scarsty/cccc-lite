#include "NetCifa.h"

namespace woco
{

NetCifa::NetCifa()
{
    registerFunctions();
}

NetCifa& NetCifa::getThis(cifa::Cifa& c)
{
    return *(NetCifa*)c.get_user_data("__this");
}

cifa::Object NetCifa::registerMatrix(cifa::Cifa& c, Matrix& m)
{
    static int count;
    cifa::Object o(count++, "Matrix");
    getThis(c).map_matrix_[o] = std::move(m);
    return o;
}

cifa::Object NetCifa::registerLoss(cifa::Cifa& c, MatrixOperator::Queue& loss)
{
    static int count;
    cifa::Object o(count++, "Loss");
    getThis(c).map_loss_[o] = std::move(loss);
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

Matrix& NetCifa::toMatrix(cifa::Cifa& c,const  cifa::Object& o)
{
    return getThis(c).map_matrix_[o];
}

MatrixOperator::Queue& NetCifa::toLoss(cifa::Cifa& c,const  cifa::Object& o)
{
    return getThis(c).map_loss_[o];
}

//void NetCifa::pushMatrix(cifa::Cifa& c, Matrix& m)
//{
//    //registerMatrix(c, m).c_str());
//}
//
//void NetCifa::pushLoss(cifa::Cifa& c, MatrixOperator::Queue& loss)
//{
//    //(L, registerLoss(L, loss).c_str());
//}

bool NetCifa::isMatrix(cifa::Cifa& c, const cifa::Object& o)
{
    return o.type == "Matrix";
}

bool NetCifa::isLoss(cifa::Cifa& c,const cifa::Object& o)
{
    return o.type == "Loss";
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
    cifa_.register_function("Matrix", [](cifa::Cifa& c, cifa::ObjectVector& v) -> cifa::Object
        {
            std::vector<int> dim;
            for (int i = 0; i < v.size(); i++)
            {
                dim.push_back(v[i]);
            }
            Matrix m(dim);
            registerMatrix(c, m);
            return 1;
        });
    cifa_.register_function("print_message", [](cifa::Cifa& c, cifa::ObjectVector& v) -> cifa::Object
        {
            fprintf(stdout, "%g\n", v[0].value);
            toMatrix(c, v[0]).message();
            return 0;
        });
    cifa_.register_function("setXYA", [](cifa::Cifa& c, cifa::ObjectVector& v) -> cifa::Object
        {
            getThis(c).setXYA(toMatrix(c, v[0]), toMatrix(c, v[1]), toMatrix(c, v[2]));
            return 0;
        });
    cifa_.register_function("clearWeight", [](cifa::Cifa& c, cifa::ObjectVector& v) -> cifa::Object
        {
            getThis(c).weights_.clear();
            return 0;
        });
    cifa_.register_function("addWeight", [](cifa::Cifa& c, cifa::ObjectVector& v) -> cifa::Object
        {
            for (auto& o : v)
            {
                getThis(c).weights_.push_back(toMatrix(c, o));
            }
            return 0;
        });

    cifa_.user_add = [](cifa::Cifa& c, const cifa::Object& l, const cifa::Object& r) -> cifa::Object
    {
        if (isMatrix(c, l) && isMatrix(c, r))
        {
            Matrix m = toMatrix(c, l) + toMatrix(c, r);
            registerMatrix(c, m);
            return 1;
        }
    };
    cifa_.user_mul = [](cifa::Cifa& c, const cifa::Object& l, const cifa::Object& r) -> cifa::Object
    {
        if (isMatrix(c, l) && isMatrix(c, r))
        {
            Matrix m = toMatrix(c, l) * toMatrix(c, r);
            registerMatrix(c, m);
            return 1;
        }
    };
    cifa_.register_function("conv", [](cifa::Cifa& c, cifa::ObjectVector& v) -> cifa::Object
        {
            auto stride = getVector(v, 2);
            auto padding = getVector(v, 3);
            Matrix m = conv(toMatrix(c, v[0]), toMatrix(c, v[1]), stride, padding);
            registerMatrix(c, m);
            return 1;
        });
    cifa_.register_function("maxpool", [](cifa::Cifa& c, cifa::ObjectVector& v) -> cifa::Object
        {
            auto window = getVector(v, 1);
            auto stride = getVector(v, 2);
            auto padding = getVector(v, 3);
            Matrix m = maxpool(toMatrix(c, v[0]), window, stride, padding);
            registerMatrix(c, m);
            return 1;
        });
    cifa_.register_function("getRow", [](cifa::Cifa& c, cifa::ObjectVector& v) -> cifa::Object
        {
            cifa::Object o(toMatrix(c, v[0]).getRow());
            return o;
        });

#define REGISTER(func) \
    cifa_.register_function(#func, [](cifa::Cifa& c, cifa::ObjectVector& v) -> cifa::Object { \
        Matrix m = func(toMatrix(c, v[0])); \
        registerMatrix(c, m); \
        return 1; \
    })

    REGISTER(relu);
    REGISTER(sigmoid);
    REGISTER(softmax);
    REGISTER(softmax_ce);

    //    cifa_.register_function("addLoss", [](cifa::Cifa& c,cifa::ObjectVector& v) -> cifa::Object
    //        {
    //            int n = (c);
    //            for (int i = 0; i < n; i++)
    //            {
    //                getThis(c)->loss_ = getThis(c)->loss_ + toLoss(c, i + 1);
    //            }
    //            return 0;
    //        });
    //    cifa_.register_function("crossEntropy", [](cifa::Cifa& c,cifa::ObjectVector& v) -> cifa::Object
    //        {
    //            auto q = crossEntropy(toMatrix(L, 1), toMatrix(L, 2));
    //            pushLoss(c, q);
    //            return 1;
    //        });
    //    cifa_.register_function("L2", [](cifa::Cifa& c,cifa::ObjectVector& v) -> cifa::Object
    //        {
    //            auto q = L2(toMatrix(c, v[0]));
    //            pushLoss(c, q);
    //            return 1;
    //        });

    return 0;
}

}    // namespace woco