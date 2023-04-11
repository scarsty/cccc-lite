# CCCC
<img src='https://raw.githubusercontent.com/scarsty/neural-demo/master/logo.png'>

## 简介

CCCC是同时支持第一代基于层和第二代基于操作的神经网络工具包，但是安装和使用远比同类工具简单。

除Linux之外，CCCC正式支持Windows，且以Windows为主要开发平台。

CCCC的功能并不及其他的开源平台，其特色是简单的配置和单机下的速度，请酌情使用。

CCCC的设计同时支持动态图和静态图，目前只有静态图形式，动态图需少量修改代码。

## 编译说明

### Windows下编译

- 任何支持C++17以上的Visual Studio和可以相互配合的CUDA均可以使用，建议使用较新的版本。核心库默认使用v140编译，可以手动升级整个工程或者安装v140编译工具。
- 下载cuDNN的开发包，将h文件，lib文件和dll文件复制到cuda工具箱目录中的include，lib/x64和bin目录。或者复制到自己指定的某个目录也可以，但是可能需要自己设置环境变量。
- 检查环境变量CUDA_PATH的值，通常应该是“C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vxx.x”（后面的数字为版本号，有可能不同）。
- 在cccc-cuda.vcproj文件中，有两处涉及cuda版本号，类似“CUDA xx.x.targets/props”，需按照自己的实际情况手动修改。
- 需要安装线性代数库OpenBLAS（推荐使用vcpkg或者msys2），将其库文件置于链接器可以找到的位置。
- 需安装libfmt，推荐使用vcpkg安装。
- 需要nvml库和nvml.dll，请使用显卡驱动中自带的，通常dll文件位于“C:\Program Files\NVIDIA Corporation\NVSMI”或者“C:\Windows\system32”，该目录有可能不在环境变量中，请自行设置或将dll文件复制出来。
- 一些dll文件默认情况并不在PATH环境变量中，应手动复制到work目录或者PATH环境变量中的目录，包括：openblas.dll，nvml.dll等。
- 下载MNIST的文件解压后，放入work/mnist目录，文件名应为：t10k-images.idx3-ubyte，t10k-labels.idx1-ubyte，train-images.idx3-ubyte，train-labels.idx1-ubyte。某些解压软件可能解压后中间的.会变成-，请自行修改。
- 编译Visual Studio工程，如只需要核心功能，请仅编译will-windows。执行以下命令测试效果，正常情况下准确率应在99%以上。
  ```shell
  will-windows -c mnist-cv.ini
  ```
- 也可以使用FashionMNIST来测试，通过mnist_path选项可以设置不同的文件所在目录。通常测试集上准确率可以到91%左右。

### Python模块

- 安装swig（建议使用msys2），放在环境变量的可执行目录，或者放在python目录。
- 安装python 3，这里我直接使用了Anaconda。
- 修改python工程中头文件和链接库文件的目录，通常是%USERPROFILE%\Anaconda3。也可以直接设置环境变量CONDA_PATH。
- 仅能编译Release版。
- 编译成功之后，用work目录下的start_jupyter.bat执行范例。
- 打开浏览器，从文档中复制代码运行。

### Linux下编译

#### x86_64
- 请自行安装和配置libfmt、CUDA和OpenBLAS，尽量使用系统提供的包管理器自动安装的版本。随安装的版本不同，有可能需要修改cmake文件。
- CUDA的默认安装文件夹应该是/usr/local/cuda，但是一些Linux发行版可能会安装至其他目录，这时需要修改CMakeLists.txt中的包含目录和链接目录。其中包含有stubs的是nvidia-ml的目录，通常不包含在默认的库目录中，可能需要自行修改。
- 下载cuDNN，放到/usr/local/cuda，注意lib的目录有可能含有64。
- 在neural目录下执行```cmake .```生成Makefile。
- ```make```编译，可以加上-j加快速度。
- 生成的可执行文件在bin文件夹。
- 推荐自行建立一个专用于编译的目录，例如：
```shell
mkdir build
cd build
cmake ..
make
```
- 在work目录下有一个范例脚本，可以用来直接编译出所有的组件，建议参考。
- 扩展和图片处理部分需要单独编译。

#### arm（测试过TK1、TX1和TX2）
- TK1只能使用cuDNN 2.0，cuda_hack.h和cuda_hack.cpp中通过一些手段将原本的cuDNN 5.0的写法转为cuDNN 2.0可以使用的形式。仅支持池化、卷积、全连接，激活函数的类型也有限。
- TX1和TX2可以使用较新的CUDA版本，编译方法基本一致。

### neural-demo
- 演示的神经网络，严格以面向对象方法构造，计算速度较慢，仅适于参考反向传播的原理。已停止更新，分离至<https://github.com/scarsty/neural-demo>。
- 所有部分都直接计算，实际上应当使用矩阵计算来优化。

### 代码格式

本工程使用修改过的clang-format进行格式设置。

### mlcc库

本工程依赖作者编写的一个公共的功能库，请从<https://github.com/scarsty/mlcc>获取最新版本，并将其置于与本目录（cccc-lite）同级的路径下。

### logo

logo由Dr. Cheng ZD设计。

旧版logo由Mr. Zou YB设计。

### 关于lite版
 
lite版只支持几个基本的激活函数和卷积、池化等基本连接。仅能使用一张显卡训练。
