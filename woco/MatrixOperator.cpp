#include "MatrixOperator.h"
#include <map>

namespace woco
{

MatrixOperator::Queue MatrixOperator::op_queue_;
int MatrixOperator::calc_ = 1;
int MatrixOperator::making_ = 0;

void MatrixOperator::beginMaking()
{
    calc_ = 0;
    making_ = 1;
    op_queue_.clear();
}

void MatrixOperator::endMaking()
{
    calc_ = 1;
    making_ = 0;
}

void MatrixOperator::forward(MatrixOperator::Queue& op_queue)
{
    for (auto& op : op_queue)
    {
        op.forward();
        //fprintf(stdout, "%d, %d\n", i, op_queue[i].type_);
        //op.matrix_.back().message();
    }
}

void MatrixOperator::backward(MatrixOperator::Queue& op_queue)
{
    //使用一个workspace减少调用次数
    /*
    if (workspace.getDataSize() == 0)
    {
        std::map<Matrix*, int64_t> map_size;
        for (auto& op : op_queue)
        {
            for (auto& m : op.matrix_in_)
            {
                map_size[&m.DMatrix()] =std::max( map_size[&m.DMatrix()], m.getDataSize());
            }
            for (auto& m : op.matrix_out_)
            {
                map_size[&m.DMatrix()] = std::max(map_size[&m.DMatrix()], m.getDataSize());
            }
        }
        for (auto& op : loss)
        {
            for (auto& m : op.matrix_in_)
            {
                map_size[&m.DMatrix()] =std::max( map_size[&m.DMatrix()], m.getDataSize());
            }
            for (auto& m : op.matrix_out_)
            {
                map_size[&m.DMatrix()] = std::max(map_size[&m.DMatrix()], m.getDataSize());
            }
        }
        int total_size = 0;
        for (auto m : map_size)
        {
            total_size += m.second;
        }
        workspace.resize(1, total_size);
        total_size = 0;
        for (auto m : map_size)
        {
            m.first->shareData(workspace, 0, 0, 0, total_size);
            total_size += m.second;
        }
    }
    workspace.initData(0);    //这个好像比较慢，采用操作模式无法避免，与层模式相比属于性能瓶颈之一
    //上面这个合并应该是不对，但是哪里不对还不知道
    */

    for (auto& op : op_queue)
    {
        op.data_weight_ = 0;
    }

    //用迭代器貌似会快
    for (auto it = op_queue.rbegin(); it != op_queue.rend(); ++it)
    {
        it->backward();
    }
}

void MatrixOperator::calloss(Queue& loss)
{
    for (auto& op : loss)
    {
        op.data_weight_ = 0;
    }
    for (auto& op : loss)
    {
        op.backward();
    }
}

//此处的宏在其后很快解除定义了
#define A (matrix_in_[0])
#define B (matrix_in_[1])
#define bias (matrix_in_[1])
#define R (matrix_out_[0])
#define W (matrix_in_[1])
void MatrixOperator::forward()
{
    switch (type_)
    {
    case MatrixOpType::ADD:
        Matrix::add(A, B, R, para_real_[0], para_real_[1]);
        break;
    case MatrixOpType::MUL:
        Matrix::mul(A, B, R, para_real_[0]);
        break;
    case MatrixOpType::ELE_MUL:
        Matrix::elementMul(A, B, R, para_real_[0]);
        break;
    case MatrixOpType::ADD_BIAS:
        MatrixExtend::addBias(A, bias, R, para_real_[0], para_real_[1]);
        break;
    case MatrixOpType::CONCAT:
        MatrixExtend::concatByChannel(matrix_in_, R);
        break;
    case MatrixOpType::ACTIVE:
        MatrixExtend::activeForward2(A, R, ActiveFunctionType(para_int_.back()), para_int_, para_real_, para_matrix_);
        break;
    case MatrixOpType::POOL:
        MatrixExtend::poolingForward(A, R, PoolingType(para_int_.back()),
            para_int2_[0], para_int2_[1], para_int2_[2], para_real_[0]);
        break;
    case MatrixOpType::CONV:
        MatrixExtend::convolutionForward(A, W, R,
            para_matrix_[0], para_int_[1], para_int2_[0], para_int2_[1], para_real_[0]);
        break;
    }
}

void MatrixOperator::backward()
{
    //若反向过程需更新多个矩阵，则在函数内部判断needUpdate
    switch (type_)
    {
    case MatrixOpType::ADD:
        if (A.needReverse()) { Matrix::add(A.DMatrix(), R.DMatrix(), A.DMatrix(), data_weight_, para_real_[0]); }
        if (B.needReverse()) { Matrix::add(B.DMatrix(), R.DMatrix(), B.DMatrix(), data_weight_, para_real_[1]); }
        break;
    case MatrixOpType::MUL:
        if (A.needReverse()) { Matrix::mul(R.DMatrix(), B, A.DMatrix(), para_real_[0], data_weight_, MATRIX_NO_TRANS, MATRIX_TRANS); }
        if (B.needReverse()) { Matrix::mul(A, R.DMatrix(), B.DMatrix(), para_real_[0], data_weight_, MATRIX_TRANS, MATRIX_NO_TRANS); }
        break;
    case MatrixOpType::ELE_MUL:
        if (A.needReverse()) { Matrix::elementMul(R.DMatrix(), B, A.DMatrix(), para_real_[0], data_weight_); }
        if (B.needReverse()) { Matrix::elementMul(R.DMatrix(), A, B.DMatrix(), para_real_[0], data_weight_); }
        break;
    case MatrixOpType::ADD_BIAS:
        MatrixExtend::addBiasBackward(A, bias, R, para_real_[0], para_real_[1] * data_weight_);
        break;
    case MatrixOpType::CONCAT:
        MatrixExtend::concatByChannelBackward(matrix_in_, R);
        break;
    case MatrixOpType::ACTIVE:
        if (A.needReverse())
        {
            MatrixExtend::activeBackward2(A, R, ActiveFunctionType(para_int_.back()), para_int_, para_real_, para_matrix_);
        }
        break;
    case MatrixOpType::POOL:
        if (A.needReverse())
        {
            MatrixExtend::poolingBackward(A, R, PoolingType(para_int_.back()),
                para_int2_[0], para_int2_[1], para_int2_[2], para_real_[0], data_weight_);
        }
        break;
    case MatrixOpType::CONV:
        MatrixExtend::convolutionBackward(A, W, R,
            para_matrix_[1], para_int_[2], para_int_[3], para_int2_[0], para_int2_[1], para_real_[0], data_weight_);
        break;
    case MatrixOpType::LOSS:
        if (scale_ != 0)
        {
            //此处直接相减，表示欧氏距离平方，若配合前一层的softmax_ce或sigmoid_ce则表示交叉熵
            Matrix::add(A, B, A.DMatrix(), scale_, -scale_, data_weight_);
        }
        break;
    case MatrixOpType::L2:
        if (scale_ != 0)
        {
            Matrix::add(A.DMatrix(), A, A.DMatrix(), data_weight_, scale_);
        }
        break;
    }
    weight_weight_ = 1;
}
#undef A
#undef B
#undef bias
#undef R
#undef W

void MatrixOperator::print(const MatrixOperator::Queue& op_queue)
{
    fprintf(stdout, "begin->");
    for (const auto& op : op_queue)
    {
        op.print();
    }
    fprintf(stdout, "end\n");
}

void MatrixOperator::print() const
{
    std::vector<std::string> strs = {
        "none",
        "add",
        "mul",
        "ele_mul",
        "add_bias",
        "concat",
        "active",
        "pool",
        "conv",
        "reshape",
        "loss",
        "l2",
    };

    fprintf(stdout, "%s->", strs[int(type_)].c_str());
#ifdef _DEBUG
    //fprintf(stdout, "\n");
    //for (const auto& m : matrix_in_)
    //{
    //    m.message();
    //}
    //for (const auto& m : matrix_out_)
    //{
    //    m.message();
    //}
    //fprintf(stdout, "\n");
#endif
}

void MatrixOperator::simpleQueue(MatrixOperator::Queue& op_queue, const Matrix& X, const Matrix& A)
{
    std::vector<int> connect_X(op_queue.size(), 0), connect_A(op_queue.size(), 0);    //1-有连接

    std::function<void(const Matrix&, int, std::vector<int>&)> check_connect = [&op_queue, &check_connect](const Matrix& X, int direct, std::vector<int>& connect)
    {
        for (int i = 0; i < op_queue.size(); i++)
        {
            if (connect[i] != 0)
            {
                continue;
            }
            auto& op = op_queue[i];
            std::vector<Matrix>*v1, *v2;
            if (direct > 0)
            {
                v1 = &op.matrix_in_;
                v2 = &op.matrix_out_;
            }
            else
            {
                v1 = &op.matrix_out_;
                v2 = &op.matrix_in_;
            }
            for (auto& m : *v1)
            {
                if (m.getDataPointer() == X.getDataPointer())
                {
                    connect[i]++;
                    for (auto& m : *v2)
                    {
                        check_connect(m, direct, connect);
                    }
                    break;
                }
            }
        }
    };

    check_connect(X, 1, connect_X);
    check_connect(A, -1, connect_A);

    int i = 0;
    for (auto it = op_queue.begin(); it != op_queue.end();)
    {
        if (connect_X[i] == 0 || connect_A[i] == 0)
        {
            it = op_queue.erase(it);
        }
        else
        {
            ++it;
        }
        i++;
    }
}

void MatrixOperator::setCalc(int c)
{
    calc_ = c;
}

MatrixOperator::Queue& MatrixOperator::getQueue()
{
    return op_queue_;
}

void getDefaultStridePadding(MatrixOpType type, const std::vector<int>& dim, std::vector<int>& stride, std::vector<int>& padding)
{
    if (type == MatrixOpType::CONV)
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

void scale(const Matrix& A, Matrix& R, real r)
{
    if (MatrixOperator::calc_)
    {
        Matrix::copyData(A, R);
        R.scale(r);
    }
    if (MatrixOperator::making_)
    {
        MatrixOperator::op_queue_.emplace_back(MatrixOperator(MatrixOpType::MUL, { A }, { R }, {}, { r }));
    }
}

void mul(const Matrix& A, const Matrix& B, Matrix& R, real a)
{
    if (MatrixOperator::calc_)
    {
        Matrix::mul(A, B, R, a);
    }
    if (MatrixOperator::making_)
    {
        MatrixOperator::op_queue_.emplace_back(MatrixOperator(MatrixOpType::MUL, { A, B }, { R }, {}, { a }));
    }
}

void elementMul(const Matrix& A, const Matrix& B, Matrix& R, real a)
{
    if (MatrixOperator::calc_)
    {
        Matrix::elementMul(A, B, R, a);
    }
    if (MatrixOperator::making_)
    {
        MatrixOperator::op_queue_.emplace_back(MatrixOperator(MatrixOpType::ELE_MUL, { A, B }, { R }, {}, { a }));
    }
}

void add(const Matrix& A, const Matrix& B, Matrix& R, realc a, realc b)
{
    if (B.number() == 1)    //此处判断需完善
    {
        addBias(A, B, R);
        return;
    }
    if (MatrixOperator::calc_)
    {
        Matrix::add(A, B, R, a, b);
    }
    if (MatrixOperator::making_)
    {
        MatrixOperator::op_queue_.emplace_back(MatrixOperator(MatrixOpType::ADD, { A, B }, { R }, {}, { a, b }));
    }
}

void addBias(const Matrix& A, const Matrix& bias, Matrix& R, realc a, realc b)
{
    if (MatrixOperator::calc_)
    {
        MatrixExtend::addBias(A, bias, R, a, b);
    }
    if (MatrixOperator::making_)
    {
        //Matrix as_1(A.getNumber(), 1);
        //as_1.initData(1);
        //需要注意cudnn自带的只支持到5维，若需更多维可以在这里修改写入op_queue的矩阵的维度
        MatrixOperator::op_queue_.emplace_back(MatrixOperator(MatrixOpType::ADD_BIAS, { A, bias }, { R }, {}, { a, b }));
    }
}

void concat(const std::vector<Matrix>& A_vector, Matrix& R)
{
    if (MatrixOperator::calc_)
    {
        MatrixExtend::concatByChannel(A_vector, R);
    }
    if (MatrixOperator::making_)
    {
        MatrixOperator::op_queue_.emplace_back(MatrixOperator(MatrixOpType::MUL, A_vector, { R }));
    }
}

void active(const Matrix& A, Matrix& R, ActiveFunctionType af)
{
    if (MatrixOperator::calc_)
    {
        MatrixExtend::activeForward(A, R, af);
    }
    if (MatrixOperator::making_)
    {
        std::vector<int> int_vector;
        std::vector<real> real_vector;
        std::vector<Matrix> matrix_vector;
        MatrixExtend::activeBufferInit(A, af, int_vector, matrix_vector);
        int_vector.push_back(af);
        if (af != 1)
        {
            af = af;
        }
        MatrixOperator::op_queue_.emplace_back(MatrixOperator(MatrixOpType::ACTIVE, { A }, { R }, int_vector, real_vector, matrix_vector));
    }
}

void active(const Matrix& A, Matrix& R, ActiveFunctionType af, std::vector<int>& int_vector, std::vector<real>& real_vector, std::vector<Matrix>& matrix_vector)
{
    if (MatrixOperator::calc_)
    {
        MatrixExtend::activeForward2(A, R, af, int_vector, real_vector, matrix_vector);
    }
    if (MatrixOperator::making_)
    {
        auto v = int_vector;
        MatrixExtend::activeBufferInit(A, af, int_vector, matrix_vector);
        v.push_back(af);
        MatrixOperator::op_queue_.emplace_back(MatrixOperator(MatrixOpType::ACTIVE, { A }, { R }, v, real_vector, matrix_vector));
    }
}

void pool(const Matrix& A, Matrix& R, PoolingType pooling_type, const std::vector<int>& window, const std::vector<int>& stride, const std::vector<int>& padding, realc a)
{
    if (MatrixOperator::calc_)
    {
        MatrixExtend::poolingForward(A, R, pooling_type, window, stride, padding, a);
    }
    if (MatrixOperator::making_)
    {
        std::vector<int> v = { int(window.size()), int(pooling_type) };
        MatrixOperator::op_queue_.emplace_back(MatrixOperator(MatrixOpType::POOL, { A }, { R }, v, { a }, {}, { window, stride, padding }));
    }
}

void conv(const Matrix& A, const Matrix& W, Matrix& R, const std::vector<int>& stride, const std::vector<int>& padding, realc a)
{
    if (MatrixOperator::calc_)
    {
        //这里好像无法记录下来工作空间，如有必要就搞个map，或者使用无需工作空间的形式
        Matrix workspace, bias;
        int method;
        MatrixExtend::convolutionForward(A, W, R, workspace, method, stride, padding, a);
    }
    if (MatrixOperator::making_)
    {
        std::vector<int> v = { int(stride.size()), -1, -1, -1 };
        MatrixOperator::op_queue_.emplace_back(MatrixOperator(MatrixOpType::CONV, { A, W }, { R }, v, { a }, std::vector<Matrix>{ 2 }, { stride, padding }));
    }
}

void reshape(const Matrix& A, Matrix& R, std::vector<int>& dim)
{
    //此处需强制建立R与A的替身关系
    R = A;
    R.resize(dim);
    if (MatrixOperator::making_)
    {
        MatrixOperator::op_queue_.emplace_back(MatrixOperator(MatrixOpType::RESHAPE, { A }, { R }));
    }
}

Matrix scale(const Matrix& A, real r)
{
    Matrix R(A.getDim());
    scale(A, R, r);
    return R;
}

Matrix mul(const Matrix& A, const Matrix& B, real a)
{
    Matrix R(A.row(), B.col(), A.getDeviceType());
    mul(A, B, R, a);
    return R;
}

Matrix elementMul(const Matrix& A, const Matrix& B, real a)
{
    Matrix R(A.getDim(), A.getDeviceType());
    elementMul(A, B, R, a);
    return R;
}

Matrix add(const Matrix& A, const Matrix& B, realc a, realc b)
{
    Matrix R(A.getDim(), A.getDeviceType());
    add(A, B, R, a, b);
    return R;
}

Matrix addBias(const Matrix& A, const Matrix& bias, realc a, realc b)
{
    Matrix R(A.getDim(), A.getDeviceType());
    addBias(A, bias, R, a, b);
    return R;
}

Matrix active(const Matrix& A, ActiveFunctionType af)
{
    Matrix R(A.getDim());
    active(A, R, af);
    return R;
}

Matrix active(const Matrix& A, ActiveFunctionType af, std::vector<int>& int_vector, std::vector<real>& real_vector, std::vector<Matrix>& matrix_vector)
{
    Matrix R(A.getDim());
    active(A, R, af, int_vector, real_vector, matrix_vector);
    return R;
}

Matrix pool(const Matrix& A, PoolingType pooling_type, const std::vector<int>& window, std::vector<int> stride, std::vector<int> padding, realc a)
{
    auto dim = A.getDim();
    getDefaultStridePadding(MatrixOpType::POOL, window, stride, padding);
    for (int i = 0; i < dim.size() - 2; i++)
    {
        dim[i] = (dim[i] + 2 * padding[i] - window[i]) / stride[i] + 1;
    }
    Matrix R(dim);
    pool(A, R, pooling_type, window, stride, padding, a);
    return R;
}

Matrix conv(const Matrix& A, const Matrix& W, std::vector<int> stride, std::vector<int> padding, realc a)
{
    auto dim = A.getDim();
    getDefaultStridePadding(MatrixOpType::CONV, W.getDim(), stride, padding);
    for (int i = 0; i < dim.size() - 2; i++)
    {
        dim[i] = (dim[i] + 2 * padding[i] - W.getDim()[i]) / stride[i] + 1;
    }
    dim[dim.size() - 2] = W.getDim().back();
    Matrix R(dim);
    conv(A, W, R, stride, padding, a);
    return R;
}

Matrix maxpool(const Matrix& A, const std::vector<int>& window, std::vector<int> stride, std::vector<int> padding, realc a)
{
    return pool(A, POOLING_MAX, window, stride, padding, a);
}

Matrix relu(const Matrix& A)
{
    return active(A, ACTIVE_FUNCTION_RELU);
}

Matrix sigmoid(const Matrix& A)
{
    return active(A, ACTIVE_FUNCTION_SIGMOID);
}

Matrix softmax(const Matrix& A)
{
    return active(A, ACTIVE_FUNCTION_SOFTMAX);
}

Matrix softmax_ce(const Matrix& A)
{
    return active(A, ACTIVE_FUNCTION_SOFTMAX_CE);
}

Matrix reshape(const Matrix& A, std::vector<int>& dim)
{
    auto R = A;
    reshape(A, R, dim);
    return R;
}

Matrix operator+(const Matrix& A, const Matrix& B)
{
    Matrix R(A.getDim(), A.getDeviceType());
    add(A, B, R);
    return R;
}

Matrix operator-(const Matrix& A, const Matrix& B)
{
    return add(A, B, 1, -1);
}

Matrix operator*(const Matrix& A, const Matrix& B)
{
    return mul(A, B);
}

Matrix operator*(real r, const Matrix& A)
{
    return scale(A, r);
}

Matrix operator*(const Matrix& A, real r)
{
    return scale(A, r);
}

MatrixOperator::Queue operator+(const MatrixOperator::Queue& A, const MatrixOperator::Queue& B)
{
    auto R = A;
    R.insert(R.end(), B.begin(), B.end());
    return R;
}

MatrixOperator::Queue& operator+=(MatrixOperator::Queue& A, const MatrixOperator::Queue& B)
{
    A.insert(A.end(), B.begin(), B.end());
    return A;
}

MatrixOperator::Queue operator*(const MatrixOperator::Queue& A, double v)
{
    auto R = A;
    for (auto& R1 : R)
    {
        R1.scale_ *= v;
    }
    return R;
}

MatrixOperator::Queue operator*(double v, const MatrixOperator::Queue& A)
{
    auto R = A;
    for (auto& R1 : R)
    {
        R1.scale_ *= v;
    }
    return R;
}

MatrixOperator::Queue crossEntropy(const Matrix& A, const Matrix& Y)
{
    MatrixOperator op(MatrixOpType::LOSS, { A, Y }, {});
    if (MatrixOperator::calc_)
    {
        Matrix R(A.getDim());
        Matrix::crossEntropy(A, Y, R);
        op.value_ = R.sum();
    }
    return { op };
}

MatrixOperator::Queue L2(const Matrix& A)
{
    MatrixOperator op(MatrixOpType::L2, { A }, {});
    if (MatrixOperator::calc_)
    {
        op.value_ = Matrix::dot(A, A);
    }
    return { op };
}

MatrixOperator::Queue L2(const std::vector<Matrix>& v)
{
    MatrixOperator::Queue q;
    for (auto& m : v)
    {
        q.emplace_back(MatrixOperator(MatrixOpType::L2, { m }, {}));
    }
    return q;
}

}    // namespace woco