# CCCC
<img src='https://raw.githubusercontent.com/scarsty/neural-demo/master/logo.png'>

cccc为libwill的核心部分，负责神经网络的训练与推理。

英文代号“will”意为“心愿”。

## 简介

cccc是同时支持第一代基于层和第二代基于操作的神经网络工具包，但是安装和使用远比同类工具简单。

cccc正式支持Windows，且以Windows为主要开发平台。可以在不需要nccl的情况下进行并行。

cccc的设计同时支持动态图和静态图。

cccc同时支持N卡和A卡，无需为不同的显卡平台编译两个版本。甚至可以在同一电脑上安装两种显卡并行计算（并不是建议你这样做）。

cccc的功能并不及其他的开源平台，其特色是简单的配置和单机下的速度，请酌情使用。

## 编译说明

### Windows下编译

- 任何支持C++20以上的Visual Studio和可以相互配合的CUDA均可以使用，建议使用较新的版本。
- 下载cuDNN的开发包，将h文件，lib文件和dll文件复制到cuda工具箱目录中的include，lib/x64和bin目录。或者复制到自己指定的某个目录也可以，但是可能需要自己设置环境变量。
- 检查环境变量CUDA_PATH的值，通常应该是“C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vxx.x”（后面的数字为版本号，有可能不同）。
- 在cccc-cuda.vcproj文件中，有两处涉及cuda版本号，类似“CUDA xx.x.targets/props”，需按照自己的实际情况手动修改。
- 需要安装线性代数库OpenBLAS（推荐使用vcpkg或者msys2），将其库文件置于链接器可以找到的位置。
- 如需支持AMD的显卡，则应下载AMD的相关开发包并安装，同时检查gpu_lib.h中对应的宏。注意miopen仅有非官方编译的Windows版本，目前计算卷积的速度很慢。
- 查看gpu_lib.h开头的ENABLE_CUDA和ENABLE_HIP的设置，修改此处或配置中的预处理部分打开或关闭对应的平台。
- 一些dll文件默认情况并不在PATH环境变量中，应手动复制到work目录或者PATH环境变量中的目录，包括openblas.dll等。
- 下载MNIST的文件解压后，放入work/mnist目录，文件名应为：t10k-images.idx3-ubyte，t10k-labels.idx1-ubyte，train-images.idx3-ubyte，train-labels.idx1-ubyte。某些解压软件可能解压后中间的.会变成-，请自行修改。
- 编译Visual Studio工程，如只需要核心功能，请仅编译cccc-windows。执行以下命令测试效果，正常情况下准确率应在99%以上。
  ```shell
  cccc-windows -c mnist-lenet.ini
  ```
- 也可以使用FashionMNIST来测试，通过mnist_path选项可以设置不同的文件所在目录。通常测试集上准确率可以到91%左右。

### Linux下编译

#### x86_64
- 请自行安装和配置CUDA，HIP（如需要）和OpenBLAS，尽量使用系统提供的包管理器自动安装的版本。随安装的版本不同，有可能需要修改cmake文件。
- CUDA的默认安装文件夹应该是/usr/local/cuda，但是一些Linux发行版可能会安装至其他目录，这时需要修改CMakeLists.txt中的包含目录和链接目录。
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
- 扩展需要单独编译。

### mlcc

本工程依赖作者编写的一个公共的功能库，请从<https://github.com/scarsty/mlcc>获取最新版本，并将其置于与本目录（cccc-lite）同级的路径下。

### logo

logo由Dr. Cheng ZD设计。

旧版logo由Mr. Zou YB设计。

### 关于lite版
 
lite版只支持几个基本的激活函数和卷积、池化等基本连接。

仅能使用一张显卡训练。

不支持半精度。
