# WILL

will是一个神经网（深度学习）的通用工具集。

## logo

![logo](https://raw.githubusercontent.com/scarsty/neural-demo/master/logo.png)

# 架构

神经网络-神经层架构，没有显式的神经元。

## 大脑

大脑里面可以包含数个神经网，并控制数据和网络的调度。

## 神经网

神经网包含数个神经层。这些神经层有固定的计算和依赖顺序，部分网络结构可能有并列的神经层，它们的计算顺序并不一定依赖。

在连接成神经网的时候，会自动去除连接不上的神经层。

## 神经层

神经层的算法为：正向激活和反向传播。

其中反向传播之前，需要先更新代价函数。

我们首先约定一套符号（在程序中也使用这些符号）：

$x$表示输入层的输入。

$y$表示期望的输出层的输出，即监督学习中的标准答案。

$a$表示每层实际上的输出，最后一层的$a$为预测值。

$W$表示每层的线性变换矩阵。

$b$表示偏置。

$C$表示损失函数，为一个数。

$\Delta x$表示$\partial C/\partial x$（为一个向量，即损失函数对$x$中每个元素的偏导数）。$y$，$a$，$W$，$b$亦同。

以上符号，除$C$之外，如无特殊说明，小写是列向量，大写是矩阵，带有下标的为一个数。在实际的训练过程中，因为批量训练（MiniBatch）的关系，一个列向量会被扩展成含有$n$列的矩阵。但是变换矩阵是不需要扩展的，即：
$$
W(a_1,a_2,a_3,...,a_n)=(Wa_1, Wa_2, Wa_3,...,Wa_n)
$$
乘法默认指矩阵乘，元素乘（或者叫Hadamard乘）用$\odot$表示。

若$x$，$a$，$y$，$b$使用对应的大写，则表示将向量按照MiniBatch扩展为矩阵。需注意$b$的矩阵的每一列都应是完全相同的。

### 计算顺序

计算顺序为：

- 正向

1. 以上一层的$A$作为输入更新$X$。如果是输入层，没有这一步。
2. 依据$X$和激活函数更新$A$。

- 反向

1. 更新$\Delta A$。除最后一层外，均需要调用下一层的相关函数来更新，因为实际上本层是不知道跟下一层的连接方式的。如果存在多个下层，结果是所有下层返回的和。
2. 更新$\Delta X$。依据激活方式来决定。
3. 更新参数。通常是$W$和$b$，如果有其他的也会更新。首先会计算$\Delta W$和$\Delta b$，之后依据求解器的设定更新$W$和$b$。

### 神经层种类

每层首先依据上一层的$A$计算出本层的$X$，之后通过激活函数计算出本层的$A$。因此每层内部的$X$和$A$必定同维。

神经层以连接方式分类，包含以下种类：

- 无连接：没有前一层，仅用于输入层。
- 全连接：每个神经元跟前面一层的所有神经元都有连接。
- 卷积：卷积操作。
- 池化：池化操作。
- 直连：直接读取上一层的数据，一般是浅复制。通常用于连续两次激活，例如批归一化后再激活。
- 合并：有多个上层，或者合并，或者求和来计算本层数据。
- 抽取：提取上一层的一个或者多个通道。

所有层都可以包含激活函数。除了直连层之外，与上层的数据维度通常是不一样的。

所有层都可以有多个下层，这些下层获得的输入是完全一样的。

只有合并层可以有多个上层。

#### 全连接

$W$为权重矩阵，行数为本层的神经元数，列数为上层的神经元数。$b$为偏置向量，行数为本层神经元数。
$$
a^{l}=\sigma(x^l)=\sigma(W^la^{l-1}+b)
$$
全连接必须自己处理$X$和$A$，接收的是上一层的$A$，经过线性组合为本层的$X$。

#### 卷积

如果用向量表示图片，那么实际上卷积可以视为一个线性变换，但是变换矩阵应该是一个稀疏阵。
例如将一个28\*28的图片变为24\*24，卷积核为5\*5，那么对应的矩阵为784\*576， 其中有25\*576个元素是非零，而且大多数都是重复的。

卷积将上一层的$A$处理为本层的$X$，并加上偏移，激活后为本层的$A$。

#### 池化

实际上平均值池化可以理解为广义卷积，与上面的讨论类似，也可以用矩阵乘。

池化通常不包含可训练的参数。

### 激活函数

所有输出和输入维度一致的情况均视为激活函数，而不视为层的种类。因此局域亮度，批归一化均被视为激活函数。

$\sigma$是激活函数，大多数时候用Sigmoid函数，该函数的变量可以为向量或者矩阵，表示对每一个元素进行相同的操作。
$$
a=\sigma(x)=\frac{1}{1+e^{-x}}
$$
除此之外，还有一些其他类型的激活函数。

| 名称                         | 数学形式                            |
| ---------------------------- | ----------------------------------- |
| 双曲正切                     | $\tanh(x)$                          |
| 软最大化（Softmax）          | ${\exp(x_i)}/{\sum_{k}{\exp(x_k)}}$ |
| 线性整流（ReLU）             | $\max(0,x)$                         |
| 裁剪线性整流（Clipped ReLU） | $\min(\max(0,x),z)$                 |

激活函数可以为None，这时一般会做一次浅复制。

### Cost函数和交叉熵

损失函数可以用预测值和标准答案差的平方和来表示：
$$
C=-\frac{1}{2n}\sum_i(a_i-y_i)^2
$$
当最后一层的激活函数为Sigmoid或者Softmax时，交叉熵更加有效。
对于Softmax作为激活函数的分类问题，交叉熵为：
$$
C=-\frac{1}{n}\sum_i{y_ilog(a_i)}
$$
注意此时大部分的$y_i$是0。
如果以Sigmoid作为激活函数进行二分类，则应为：
$$
C=-\frac{1}{n}\sum_i[y_ilog(a_i)+(1-y_i)log(1-a_i)]
$$
注意前面的$1/2n$在实际计算时可以视为一个常数忽略。

如果使用交叉熵作为损失函数，那么最后一层的$\partial C/\partial x$大部分时候可以被简化为：
$$
\frac{\partial C}{\partial x_i}=a_i-y_i
$$
即：
$$
\Delta x=a-y
$$

#### 附：Softmax反向传播的推导

这里对如何反向推导作一个示范。

从上一层计算得到的是Cost函数$C​$对$a​$的导数，注意$C​$是一个数。

那么已知$a$，$x$以及：
$$
\frac{\partial C}{\partial a}=(\frac{\partial C}{\partial a_1},\frac{\partial C}{\partial a_2},...,\frac{\partial C}{\partial a_k})^T
$$
要得到：
$$
\frac{\partial C}{\partial x}=(\frac{\partial C}{\partial x_1},\frac{\partial C}{\partial x_2},...,\frac{\partial C}{\partial x_k})^T
$$
这里我们以第一项为例：
$$
\frac{\partial C}{\partial x_1}=\sum_{i=1}^{k}\frac{\partial C}{\partial a_i}\frac{\partial a_i}{\partial x_1}=\frac{\partial C}{\partial a_1}\frac{\partial a_1}{\partial x_1}+\sum_{i=2}^{k}\frac{\partial C}{\partial a_i}\frac{\partial a_i}{\partial x_1}
$$
将
$$
a_i=\frac{e^{x_i}}{\sum_{j=1}^{k}e^{x_j}}
$$
代入上式，得到：
$$
\frac{\partial C}{\partial x_1}=a_1(1-a_1)\frac{\partial C}{\partial a_1}-\sum_{i=2}^{k}a_1 a_i\frac{\partial C}{\partial a_i}=a_1(\frac{\partial C}{\partial a_1}-\sum_{i=1}^{k} a_i\frac{\partial C}{\partial a_i})
$$
即：
$$
\Delta x_i=a_i(\Delta a_i - a^T \Delta a)
$$
类似，对于Softmax-Log
$$
a_i=\log(\frac{e^{x_i}}{\sum_{j=1}^{k}e^{x_j}})
$$
可以得到：
$$
\Delta x_i=\Delta a_i - e^{a_i} \sum_{j=1}^k \Delta a_j
$$



### 求解器

即如何利用$\Delta W$来更新$W$的方式。
可以选用SGD、SGD+动量、NAG、AdaDelta、Adam、RMSProp等。

# 数据类型

### 浮点数

在Matrix类和所有神经网相关类中，浮点数统一使用real。如果使用模板的话，模板类的规模会过大且难以维护。

real的定义在types.h中，可能为float或者double，默认为float。如果需要计算double类型（通常要硬件支持），则应修改此处并重新编译。

```cpp
#ifndef _DOUBLE_PRECISION
#define _SINGLE_PRECISION 
typedef float real;
#else 
typedef double real;
#endif
```

神经网络中主要使用float，没有必要使用double。大部分民用显卡的double的计算能力都比较差。

因为blas函数库对float和double调用的是不同名函数，此处用两个类Cublas和Cblas将其重新封装，以同名重载的方式调用。因此在实际代码中float和double使用的是同名的函数，没有使用模板。

### 矩阵类

矩阵类的重要作用是封装CUDA运算。所有矩阵基本运算均置于类内部，不属于基本运算的置于MatrixEx类。后者为纯静态类。

所有矩阵保存的方式都是fortran风格的列优先，与cublas和cblas一致。

矩阵所使用空间是由共享指针管理，矩阵的赋值相当于生成一个完全一致的副本，与原矩阵并没有区别。对矩阵进行resize，则其所有副本会有同样的变化。

矩阵声明的时候，应指明其所在位置，即使其申请空间为0。

若一个矩阵想专用于共享数据，则一开始不应为其分配空间。

矩阵共享空间的时候，应尽量使用包含目标矩阵的函数。若直接将指针传入矩阵用于共享，则不能获知该指针所在的设备，此时应手动指明设备位置。

对共享的空间矩阵进行resize的时候需要慎重行事，因为此时并不会重新分配空间，存在越界的可能。

矩阵的反向由自身管理，在不处理反向的情况下，禁止使用。

MatrixEx计算中，凡是函数名包含backward的，均在内部处理反向数据。

#### 两种矩阵

构造函数包含数个原型。矩阵有两种形式，即普通矩阵和四阶张量。其中四阶张量的前3个维度在一起视为矩阵的一列。

矩阵的初值是无法确定的，但是保存在显存中的矩阵初值经常都是0。

在大部分情况下，矩阵会优先在显卡中创建数据指针。但是需要注意的是在输出矩阵之前，需将数据复制到内存。同样，载入矩阵会先载入到内存，再复制到显存。

#### 寻址

按照行列来寻址。注意矩阵中的元素按照行列顺序列出，与直角坐标系相反！

返回的元素大多数是引用。但是注意，引用的地址如果在显存中，是无法赋值和读取的。

# 并行

采取数据并行。

显存的占用通常包含以下内容：

- 训练数据，通常会较大。
- 每个minibatch所占用的显存，贯穿整个网络，通常会较大。
- 网络参数占用的显存，通常不会特别大。

并行的情况下，需要一个主网络。每次训练都会将网络中的参数取平均值，该过程由主网络完成。各个分网络之后会各自将这部分数据取回。

# 数据准备器

该类用来生成和变换训练数据，可以根据实际情况增加对应子类。设置位于[data_preparer]块中。

# 其他

## 通用类

详见common目录下的介绍。

## 特殊功能

以下类用于实现一些特殊功能。

### MatrixFiller

用于填充矩阵，有数种填充方法。

### GpuControl

包含cudnn/cublas/rocblas/miopen等的初始化过程和全局变量。

用户任何时候都不应自行创建该类型的对象！对象的总数必定等于系统中安装的设备数，在首次选中某个设备时会进行一次初始化。

所有矩阵都会有一个该类型的指针。

### VectorMath

一些基本计算，函数均是模板函数，大部分inline直接使用。用于CPU计算。

### cccc-cuda / cccc-hip

不包含于cublas/cuDNN（或rocblas/MIOpen）中的计算，需要自行撰写。CUDA 部分位于 cccc-cuda 项目，HIP/ROCm 部分位于 cccc-hip 项目，对应包装置于 MatrixEx 类中。

该文件使用宏简化代码，使用需注意。

### cuda_hack

用一些手段将cuda高版本的写法转为低版本的写法，用于一些无法升级软件的特殊硬件（例如TK1）。

# 使用方法

## 编译

适当版本的CUDA和cudnn需要安装好，并配置相关的环境变量。除此之外，需自行下载opencv的头文件和库文件，放在include和lib目录中。因为占用空间较大并没有加入版本控制。

python接口部分默认使用Anaconda的python库，用户可能需要手动修改这里的目录配置。

这样Visual Studio可以直接生成整个工程，如果编译过程出错，一般是调试文件放在了同一个目录造成的，这时可以尝试仅编译will-windows，再编译其他附加库。

在Linux下，可以使用包管理工具安装CUDA和opencv，再使用以下命令即可。其中CXX可以为g++、clang++、icpc。在python目录下面有一个编译脚本，可以参考。

```shell
export CXX=/usr/bin/clang++
cmake .
make -j
```

注意需要安装libopencv-dev、libopenblas-dev、以及libopenmp。其中openblas仅在CPU模式下使用。

即使使用GPU进行训练，仍有少量计算需要使用CPU，故此仍然需要openblas或者mkl等。

## 命令行

工程会编译得到libwill.lib，will-cuda.lib等库文件，同时可以得到可执行文件will-windows。在linux下也使用这个名字。

```bash
usage: will-windows [options] ... 
options:
  -c, --config        config file (ini format) of the net (string [=will.ini])
  -a, --add-config    additional config string ([sec1]key1=v1;key2=v2[sec2]key1=v1...) (string [=])
  -v, --version       version information
  -?, --help          print this message
```

在命令行中使用一个ini文件来指定网络使用的参数：

```shell
will-windows -c mnist-cv.ini
```

## 配置文件

格式为ini，一个ini文件的示例如下（LeNet）：

```ini
[will]
LoadFile = save-1.txt
SaveFile = mnist_cv_save2.txt

LoadNet = 0

Batch = 100
save_epoch=1

test_test = 1
test_train = 1
test_train_origin=1

Trainepochs = 10
OutIter = 100
Testepoch = 1

LearnRateBase = 0.01

weight_decay = 5e-4
Momentum = 0.9

lr_adjust_method = fixed

Testmax = 1
USE_CUDA = 1

solver=adadelta

output_net=1

mp=1

[data_preparer]
fill = 0
trans = 0
mnist=1
flip_transpose = 0

[data_preparer_test]
fill = 0
trans = 0
mnist=1
flip_transpose = 0

[extra_test]
fill = 0
trans = 0
mnist=0
data_in_txt=1
data_file = dig.txt

[layer_in]
type = null
data = 28, 28
channel = 1
next = layer_cv1

[layer_cv1]
type = conv
channel = 50
window = 5, 5
needbias = 1
active = none
next = layer_pl1

[layer_pl1]
type = pool
window = 2, 2
active = none
next = layer_cv2

[layer_cv2]
type = conv
channel = 50
window = 5, 5
needbias = 1
active = none
next = layer_pl2

[layer_pl2]
type = pool
window = 2, 2
active = none
next = layer_full1

[layer_full1]
type = full_connect
Node = 100
active = relu
next = layer_out
initweight = xavier

[layer_out]
type = full_connect
node = 10
active = softmax
initweight = xavier
```

ini文件中的设置均不区分大小写，也可以任意添加下划线。

### 公共部分

公共部分均写在[will]块中，指定网络使用的一些公共参数，下半部分是网络的结构。注意在选项设置中，除了next项的值之外，其他部分使用大小写，或者有无下划线都是没有影响的。

如果一些选项的参数是多于一个的，则应使用逗号隔开。例如`mp_device=0,1`。

一些重要的参数为：

| 选项名           | 说明                           | 选项   |
| ---------------- | ------------------------------ | ------ |
| train_epochs     | 训练的epoch数，默认是20次      | 整数   |
| use_cuda         | 是否使用cuda设备               | 整数   |
| mp               | 使用cuda设备的个数             | 整数   |
| mp_device        | 指定使用的cuda设备             | 整数组 |
| learn_rate_base  | 基础学习率                     | 浮点数 |
| batch            | 每批学习的数据量               | 整数   |
| weight_decay     | 正则化系数                     | 浮点数 |
| momentum         | 动量                           | 浮点数 |
| test_type        | 测试类型                       | 整数   |
| lr_adjust_method | 调整学习率方式                 | fixed  |

如果没有特别指定cuda设备，程序会首先查找可用的设备数，并依据计算能力，空余显存，当前温度以及PCI编号的距离选择一个或者多个较空闲的设备。但是这样选择出来的设备并不一定是最合理，用户有特殊需求时应手动指定。

test_type为1时，会测试最大值。为2时，是拟合图片任务，会输出结果的字符缩略图。

设备的编号与nvidia-smi所显示的顺序一致。程序运行时建议使用GPU-Z或者类似工具来监视设备的状态。例如：


```shell
watch -n 1 nvidia-smi
```

Windows下如果有此需要，推荐使用WPF来编写一个简单的监视功能（参考<https://github.com/scarsty/gpustat>），其他GUI程序可能会有闪烁现象。以下代码为范例：

```c#
namespace gpustat
{
    /// <summary>
    /// MainWindow.xaml 的交互逻辑
    /// </summary>
    /// 
    public partial class MainWindow : Window
    {
        DispatcherTimer timer;

        public MainWindow()
        {
            InitializeComponent();
            timer = new DispatcherTimer();
            timer.Tick += new EventHandler(timer_Tick);
            timer.Interval = new TimeSpan(0, 0, 1);
            timer.Start();
        }

        private void timer_Tick(object sender, EventArgs e)
        {
            System.Diagnostics.Process pProcess = new System.Diagnostics.Process();
            pProcess.StartInfo.FileName = @"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe";
            pProcess.StartInfo.UseShellExecute = false;
            pProcess.StartInfo.WindowStyle = System.Diagnostics.ProcessWindowStyle.Hidden;
            pProcess.StartInfo.RedirectStandardOutput = true;
            pProcess.StartInfo.CreateNoWindow = true;
            pProcess.Start();
            string strOutput = pProcess.StandardOutput.ReadToEnd();
            pProcess.WaitForExit();
            richTextBox.Document.Blocks.Clear();
            richTextBox.Document.Blocks.Add(new Paragraph(new Run(strOutput)));
        }
    }
} 
```

### 层设置

如果使用命令行，网络必须以`layer_in`层开始，以`layer_out`层结束，不能使用别的名字。

但是在python环境中，因为可以手动创建网络，故此时可以使用其他名字。

中间的层的名字需以```layer_```开头，不能重复。层间的连接使用`next`关键字。层的连接方式由`type`指定。在初始化网络的时候，会检查所有层的连接情况是否合理，合理的层必须与输入和输出层都有连接（此处注意连接是单向的）。这也是will的一个特点，在需要频繁修改网络结构测试的时候，无需对配置文件作很多的改动，层的实际顺序也与书写顺序无关，只与next有关。

注意某些设置是首先读取公共部分为默认值，再读取本层设置的。例如基础学习率，求解器类型等。

输入层的设置范例如下：

```ini
[layer_in]
type = null
data = 28, 28
channel = 1
next = layer_cv1
```

输入层的type必须是none或者null，也不要添加任何激活。

data表示数据的维度。channel是通道数，如果不写，则认为data的最后一位是通道数。因此以下写法与上面等价：

```ini
[layer_in]
type = null
data = 28, 28, 1
next = layer_cv1
```

此时不要再设置channel。

一个典型的卷积层设置为：

```ini
[layer_cv1]
type = conv
active = relu
window = 5,5
next = layer_pool1
```

池化层的设置与此类似。

程序会扫描所有`layer_`块，并试图将它们连接。在首次连接之后，会从两侧扫描所有层，去掉与输入层和输出层无连接的层，并再次连接。

但是这样并不能保证得到一个正确的网络，使用者在编写的时候应保证网络部分正确。

每一层都可以有任意多个下一层，但是仅有concat层可以有多个上一层。连接方式则有求和或者通道连接两种，例如：

```ini
[layer_concat1]
type = concat
concat_type = add
next = layer_cv
```

目前有以下类型的层：

| type选项名               | 说明            | 附加设置                                      |
| ------------------------ | --------------- | --------------------------------------------- |
| none, null               | 初始数据        | data：输出个数，可为多维                      |
| full_connect, full, fc   | 全连接          | node：输出个数，可为多维                      |
| convolution, conv        | 卷积            | window：窗口尺寸，stride：步长，padding：填充 |
| pooling, pool            | 池化            | window：窗口尺寸，stride：步长，padding：填充 |
| direct                   | 直连            | data：可以重整数据格式                        |
| correlation, corr, conv2 | 相关            | window：窗口尺寸，stride：步长，padding：填充 |
| combine                  | 按照channel合并 | 通常无需特别设置                              |
| extract                  | 抽取某个channel | start：起始通道，channel：通道数              |
| roteigen                 | 旋转特征        |                                               |
| norm2                    | 归一化          |                                               |
| transpose                | 转置            |                                               |
| nac                      | NAC             |                                               |

其中最后四个为实验功能，基本不会使用。

所有类型的层均包含next（下一层的名字），need_bias（是否需要偏移），

### 激活函数

激活函数的输出矩阵维度与输入矩阵的维度完全相同输出，而层连接则不是相同的。

| active选项名 | 说明                  |
| ------------ | --------------------- |
| none         | 原样                  |
| sigmoid      |                       |
| relu         | 线性整流              |
| softmax      |                       |
| sigmoid_ce   | sigmoid，反向为交叉熵 |
| softmax_ce   | softmax，反向为交叉熵 |

### 脚本形式

也可以使用一段简化的c++11风格的脚本，来定义网络结构，此时需设置主section中的cifa=1。

例如下面定义了一个等价的网络：

```c++
[cifa]
structure=
    batch=100;
    act = active_relu;
    X = Matrix(28, 28, 1, batch);
    W1 = Matrix(5, 5, 1, 50);
    b1 = Matrix(1, 1, 50, 1);
    W2 = Matrix(5, 5, 50, 50);
    b2 = Matrix(1, 1, 50, 1);
    A1 = active(maxpool(conv(X, W1) + b1, { 2, 2 }),act);
    A2 = active(maxpool(conv(A1, W2) + b2, { 2, 2 }),act);
    W3 = Matrix(256, A2.row());
    b3 = Matrix(256, 1);
    W4 = Matrix(10, 256);
    b4 = Matrix(10, 1);
    A = sigmoid_ce(W4 * active(W3 * A2 + b3,act) + b4);
    setXY(X, A);
```

也可以减少中间变量，写得更简洁：
```c++
structure=
    batch=100;
    act = active_relu;
    X = Matrix(28, 28, 1, batch);
    A1 = active(maxpool(conv(X, Matrix(5, 5, 1, 50)) + Matrix(1, 1, 50, 1), { 2, 2 }), act);
    A2 = active(maxpool(conv(A1, Matrix(5, 5, 50, 50)) + Matrix(1, 1, 50, 1), { 2, 2 }), act);
    A = softmax_ce(Matrix(10, 256) * active(Matrix(256, A2.row()) * A2 + Matrix(256, 1), act) + Matrix(10, 1));
    setXY(X, A);
```

如果需要调用ini中的其他定义，可以用section::key的形式。例如：

```c++
[train]
batch=100
cifa=1
...

[cifa]
structure = 
    ...
    batch=train::batch
    ...
```

这里需要注意的是每个函数的输入、输出和权重是固定的参数位置。

这种定义比上面的形式更为灵活，例如可以轻易实现权重、输入的复用等，但是需要注意语法的检查。

### 准备器

读取数据集需要用户自己编写C++代码实现，这部分会在主线程中运行。

用户需要编写从磁盘中读取数据到Matrix，以及一些自定义的变换等。基类中已经包含了一些简单变换，如亮度、对比度、翻转转置等。

如需要自己编写，需参考cccc-mnist的代码。

若设置mnist=1，则表示使用mnist的训练数据（60000个）。若为2，则表示使用测试数据（10000个）。

## Python接口

python接口需要预先安装swig，可以使用build.sh生成。注意vs工程中的路径可能与用户实际的配置不同，应手动修改。will.py和动态库（\_will.pyd或者\_will.so）应放在工作目录下。

一个例子如下：

```python
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np 
import will
import time
import os
import ctypes

if 'brain' in dir():
    del brain

ini_file = 'mnist-cv.ini'  
brain = will.Brain()
brain.getOption().loadIniFile(ini_file)
# here you can set options yourself
brain.init()
brain.run()
```

以上首先定义了一部分常用的功能（例如将will矩阵中的数据转图片），下面的运行部分先创建一个网络，并运行。

这里有两种方式创建网络，其一为直接通过ini文件，这时除了训练次数之外，与命令行区别不大。这里是可以载入多个ini文件的，如果有相同的项，后面的会覆盖前面的。所以可以将网络写在一个ini文件，公共部分写在另一个ini文件。或者先载入一个专门用来测试的文件（必须要载入网络参数，load_net=1），再载入一个训练用的公共部分（load_net=0覆盖前面的）。

需要注意的是，loadini是exe的特有选项，在python中是无效的。但是可以通过载入多个ini文件来实现同样的功能。

在载入ini之后，初始化之前，可以使用setOptions命令手动修改一些设置参数。当然也可以不载入option，完全手动设置。

例如设置上面提到的卷积层可以这样写：

```python
brain.getOption().setOptions('layer_conv1', 'type=conv; active = relu; window=5,5; next=layer_pool1')
```

每个选项间用分号隔开。

若设置较多，分几行也是一样的：

```python
brain.getOption().setOptions('layer_conv1', 'type=conv')
brain.getOption().setOptions('layer_conv1', 'active = relu; window=5,5; next=layer_pool1')
```

公共部分的第一个参数可以为will，或者省略第一个参数（重载），不能写成空字串。

如果网络比较深，可以写一段循环来增添层，而不是将ini文件写得很长。

如果需要将矩阵中的数据转为numpy的格式，方便后续的处理，请使用类似下面的方法：

```python
def toImg(matrix, t):
    m = matrix.getWidth()
    n = matrix.getHeight()
    c = matrix.getChannel()
    adr = int(matrix.getDataPointer(0,0,0,t))
    img = np.frombuffer((ctypes.c_float*m*n*c).from_address(adr), np.float32).reshape(c,m,n).copy()
    # note: or reshape(m,n,c,order='F')
    if c == 1:
        img = img.reshape(m,n)
    else:
        img = img.transpose(1,2,0)
    return img

def findMax(matrix, t):
    n=matrix.getChannel()
    v=[0]*n
    for i in range(0,n):
        v[i]=matrix.getDataValue(i,t)
    return v.index(max(v)),max(v)
```


注意此时矩阵必须在CPU中，如果矩阵在GPU中，则应首先复制到CPU：

```python
y_matrix = Y.clone(will.MATRIX_DATA_INSIDE, will.CUDA_CPU)
```

如果不需要转置操作，可以不用加copy。如果所画的图不是1通道，3通道和4通道其中之一，后续需要自己处理，例如将多通道并排画出为：

```python
def reshape2(img1):
    img2=img1.transpose(0,2,1).reshape(img1.shape[0],img1.shape[1]*img1.shape[2])
    return img2
```

下面是一个画图的例子：

```python
def drawk(img, plt, k, i1,i2,f):
    ax=plt.subplot(30, 6, k)  
    ax.imshow(img, norm=plt.Normalize(vmin=0,vmax=1),cmap='gray')
    ax.text(3, 26, '%d/%d(%f)' % (i1,i2,f), color='white', family='monospace', fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
  
def drawbig(data, begin):
    x_matrix = data.X()
    y_matrix = data.Y()
    a_matrix = data.A()
    plt.figure(figsize=(15, 75))          #显示图像大小
    k=0
    for i in range(begin, x_matrix.getCol()): #获取列的个数
        img1 = toImg(x_matrix, i)
        (y,yv) = findMax(y_matrix, i)
        (a,av) = findMax(a_matrix, i)
        if y==a:
            continue
        k=k+1        
        drawk(img1, plt, k, y,a,av)
        if k >= 60:
            break

brain.getNet().test("Train cpu", brain.train_data_cpu_, 0,0)
drawbig(brain.train_data_origin_,0)

```

### 网络可视化

网络的可视化也使用python来制作，需要安装graphviz，并设置相关的环境变量。

以下例程可以绘制出网络的结构：

```python
from graphviz import Graph
def net_show(net):
    connection_types_cl = ['black','green','red','blue','lightblack','orange','pink','lightred']
    connection_types = ['none','fc','conv','pool','direct','corr','combine','extract']
    activation_types = ['none','sigmoid','relu','tanh','clipped_relu','scale','softmax','softmax','softmax','findmax','dropout','recurrent','softplus','lrn','lcn','dn','bn','st']
    
    layers = []
    layers_type = {}
    layers_colors = {}

    for i in range(0, net.getLayersCount()):
        layer = net.getLayer(i)
        layer.name = layer.getName()
        layer.out_shape = [layer.getOutWidth(), layer.getOutHeight(), layer.getOutChannel()]
        layer.next_layers = []
        layer.layer_type = connection_types[layer.getConnection()]
        layer.layer_color = connection_types_cl[layer.getConnection()]
        layer.label = layer.name + ', ' + activation_types[layer.getActiveFunction()+1] + '\n'+str(layer.out_shape)
        for j in range(0, layer.getNextLayersCount()):
            layer.next_layers.append(layer.getNextLayer(j).getName())
            #print layer.name, layer.out_shape,layer.next_layers,layer.layer_type
        layers_type[layer.name] = layer.layer_type
        layers.append(layer)

    g = Graph('G', filename=ini_file+'.gv')
    g.attr('node',shape='rect',color='lightblue', fontname='Arial', fontsize='12')
    g.body.extend(['rankdir=BU', 'size="20,10"'])
    for i in range(0,len(layers)):
        layer = layers[i]
        g.attr('node',color=layer.layer_color, fontname='Arial')
        g.node(layer.name,label= layer.label )
        #print layer.name
    for i in range(0,len(layers)):
        layer = layers[i]
        for nl in layer.next_layers:
            g.attr('edge', fontname='Arial')
            g.edge(layer.name,nl,label = layers_type[nl] )
    return g

net = brain.getNet()
net_show(net)
```

用户可以将这段程序放在jupyter notebook的一个单独的块中，方便调整格式。

# extension

扩展部分可以编写为dll（或者so），并用DynamicLibrary类来载入。

这里常被用来扩展数据准备器。用户可以编写自己的数据准备器，来适应不同的需求。

# 关于cudnn的补充

## Turing架构

从cudnn的文档来看，Turing架构的支持主要是卷积计算和RNN计算，在计算之前，增加类似下面的语句即可：

```c++
cudnnSetConvolutionMathType(cuda->convolution_desc_, CUDNN_TENSOR_OP_MATH);
```

## Tensor

tensor（张量）是基本的单位，用来描述一组数据的结构。在cuDNN中，最常用的是4阶张量，即图片本身是2维（W\*H），同时有C张提取的特征图片，本次输入的数据组为N。

Will支持最高8阶的张量（cudnn的上限），存储顺序类似NCHW的方式。这里写的顺序与存储的优先级是相反的（待确定），这是fortran的习惯，沿用至BLAS。

packed表示内存连续，即一行紧接着上一行。如果两行之间还有一些无用数据，则不是packed。大部分算法要求NCHW中最后两个是packed的，而will中要求数据全部是packed的。

## 为何反向传播时需要提供正向的结果

cudnn中使用$y$和$\Delta y$表示输出量，而大部分文献和本工程中，使用的符号是$a$，请注意区别。

在反向传播的时候，需要使用$a$，$\Delta a$，$x$来计算$\Delta x$，这里注意计算公式在形式上并未使用$a$：
$$
\Delta x = \sigma'(x)\odot\Delta a
$$
但是一些情况下使用$a$会加快计算的速度，例如对于Sigmoid函数：

$$
a=\sigma(x)=\frac{1}{1+e^{-x}}
$$
其导数为
$$
\sigma'(x)=\frac{e^{-x}}{(1+e^{-x})^2}=a(1-a)
$$
可见使用$a$时避免了指数计算，速度应更快。故cudnn中的相关函数同时需要$x$和$a$作为输入参数。




