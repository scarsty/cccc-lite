#include "NetMnist.h"

namespace woco
{

void NetMnist::structure()
{
    //以下构造网络的结构，相当于执行一次前向的计算，必须指明X_，Y_的尺寸，和如何计算A_
    //因可能每个网络所在的设备不同，需将它们以赋值的形式，丢弃之前数据
    auto X = Matrix(28, 28, 1, 50);    //入口

    //此处开始相当于执行一次前向计算，在非计算模式下会仅生成计算图，并不实际计算
    //若需要在计算模式下执行实际计算，应关闭生成图的参数，将网络结构部分提取到独立函数，同时权重改为用类中的字段表示
    //此时中间变量会按照C++的规则进行生成和释放，可以节省一部分内存，但是会比较慢

    //网络开始
    Matrix W1(5, 5, 1, 50), b1(1, 1, 50, 1);
    Matrix W2(5, 5, 50, 50), b2(1, 1, 50, 1);

    auto A1 = relu(maxpool(conv(X, W1) + b1, { 2, 2 }));
    auto A2 = relu(maxpool(conv(A1, W2) + b2, { 2, 2 }));

    Matrix W3(256, A2.getRow()), b3(256, 1);
    Matrix W4(10, 256), b4(10, 1);

    auto A = softmax_ce(W4 * relu(W3 * A2 + b3) + b4);    //计算结果
    //网络结束

    auto Y = Matrix(A.getDim());    //出口

    setXYA(X, Y, A);

    //保存需要训练的参数及初始化
    weights_ = { W1, b1, W2, b2, W3, b3, W4, b4 };
    for (auto& m : weights_)
    {
        MatrixExtend::fill(m, RANDOM_FILL_XAVIER, m.getChannel(), m.getNumber());
    }

    //损失
    loss_ = crossEntropy(A, Y) + 1e-4 * (L2(W1) + L2(W2) + L2(W3) + L2(W4));

    //Matrix W1(100, 196), b1(100, 1);
    //Matrix W2(10, 100), b2(10, 1);
    //A_ = softmax_ce(W2 * relu(W1 * X_ + b1) + b2);
    //loss_ = crossEntropy(A_, Y_) + 1e-4 * (L2(W1) + L2(W2));

    //weights_ = { W1, b1, W2, b2 };
    //for (auto& m : weights_)
    //{
    //    MatrixExtend::fill(m, RANDOM_FILL_XAVIER, 10, 10);
    //}
}

}    // namespace woco