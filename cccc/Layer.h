#pragma once
#include "MatrixEx.h"
#include "MatrixOp.h"
#include "Option.h"
#include "Solver.h"
#include "Timer.h"
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace cccc
{

//神经层
//下面凡是有两个函数的，在无后缀函数中有公共部分，在带后缀函数中是各自子类的功能
class Layer
{
public:
    Layer();
    virtual ~Layer();
    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;

protected:
    int id_;
    std::string layer_name_;    //本层的名字

    Layer* prev_layer_ = nullptr;        //仅合并数据层可以有多个前层，其余层若多于一个会报警
    std::vector<Layer*> prev_layers_;    //除了合并数据层外，该向量仅用于生成网络结构，计算时无用
    std::vector<Layer*> next_layers_;    //后层可以有多个，它们获取到的是同样的数据

    LayerVisibleType visible_type_ = LAYER_VISIBLE_HIDDEN;
    LayerConnectionType connetion_type_ = LAYER_CONNECTION_NONE;    //这个字段仅为标记，实际连接方式是虚继承

    Option* option_;

public:
    //前面一些字段的设置
    void setID(int id) { id_ = id; }
    int getBatchSize();
    void setVisible(LayerVisibleType vt) { visible_type_ = vt; }
    LayerConnectionType getConnection() { return connetion_type_; }
    LayerConnectionType getConnection2();
    void setConnection(LayerConnectionType ct) { connetion_type_ = ct; }
    //void getOutputSize(int& width, int& height, int& channel) { width = out_width_; height = out_height_; channel = out_channel_; }
    const std::string& getName() { return layer_name_; }
    void setName(const std::string& name);

    void setOption(Option* op) { option_ = op; }
    //static void setEpochCount(int ec) { epoch_count_ = ec; }

public:
    void addPrevLayers(Layer* layer)
    {
        prev_layer_ = layer;
        prev_layers_.push_back(layer);
    }
    void addNextLayers(Layer* layer) { next_layers_.push_back(layer); }

    Layer* getPrevLayer() { return prev_layer_; }
    std::vector<Layer*> getPrevLayers() { return prev_layers_; }
    std::vector<Layer*> getNextLayers() { return next_layers_; }
    void clearConnect()
    {
        prev_layer_ = nullptr;
        prev_layers_.clear();
        next_layers_.clear();
    }
    int getNextLayersCount() { return next_layers_.size(); }
    Layer* getNextLayer(int i) { return next_layers_[i]; }

public:
    //这几个矩阵形式相同，计算顺序： X, A, ..., dA, dX
    //当本层没有激活函数时，A与X指向同一对象，dA与dX指向同一对象
    //MatrixSP X_;    //X收集上一层的输出
    MatrixSP A_ = makeMatrixSP();    //激活函数作用之后就是本层输出A，输入层需要直接设置A
    MatrixSP W_ = makeMatrixSP();
    MatrixSP b_ = makeMatrixSP();

public:
    void message();
    void makeMatrixOp(std::vector<MatrixOp>& op_queue);
};

}    // namespace cccc