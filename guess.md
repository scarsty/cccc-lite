# WOCO

woco是一个实验性质的工程，其思路为使用表达式，隐式生成计算图，并计算其正向和反向。

## 矩阵

为了隐式生成图的设计可以成立，矩阵或张量必须同时支持以下功能：

* 构造或者赋值实际为共享内存区域，否则计算图是非连续的。
* 必须支持返回矩阵形式的函数，以及主要算符的重载。
* 需支持cudnn中纯C风格的数据。
* 矩阵需支持共享部分数据。
* 反向需共享数据。

故此矩阵类采取了包装类，共享指针，共享指针套唯一指针等设计。

## 计算图

计算图由算子的类型，参与计算的张量，以及计算的参数组成，为有向图。

有以下数种方式生成计算图：

### C++的表达式

```c++
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

    //损失
    loss_ = crossEntropy(A, Y) + 1e-4 * (L2(W1) + L2(W2) + L2(W3) + L2(W4));
```

weights_保存了需要训练的矩阵，即权重。

### Cifa脚本

使用Cifa执行一个接近C++语法的脚本。

```c++
    auto batch = 100;
    auto X = Matrix(28, 28, 1, batch);
    auto W1 = Matrix(5, 5, 1, 50);
    auto b1 = Matrix(1, 1, 50, 1);
    auto W2 = Matrix(5, 5, 50, 50);
    auto b2 = Matrix(1, 1, 50, 1);
    auto A1 = relu(maxpool(conv(X, W1) + b1, { 2, 2 }));
    auto A2 = relu(maxpool(conv(A1, W2) + b2, { 2, 2 }));
    auto W3 = Matrix(256, A2.getRow());
    auto b3 = Matrix(256, 1);
    auto W4 = Matrix(10, 256);
    auto b4 = Matrix(10, 1);
    auto A = softmax_ce(W4 * relu(W3 * A2 + b3) + b4);
    auto Y = Matrix(1, 1, 10, batch);
    setXYA(X, Y, A);
    addWeight(W1, b1, W2, b2, W3, b3, W4, b4);
    addLoss(crossEntropy(A, Y) + 1e-4 * (L2(W1) + L2(W2) + L2(W3) + L2(W4)));
```
上述脚本可以直接复制到C++，将其编译到可执行文件中。

### 通过onnx创建

通过插件onnx插件完成。

注意支持的操作很有限。

## 损失函数

损失函数的写法比较符合数学上的定义。

特别地，L2正则化此处被认为是损失的一部分，而不是求解器的一部分。
