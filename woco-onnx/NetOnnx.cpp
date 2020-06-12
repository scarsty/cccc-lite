#include "NetOnnx.h"
#include "onnx.proto3.pb.h"
#include <fstream>

namespace woco
{

void NetOnnx::structure()
{
    onnx::ModelProto model;
    std::ifstream file("mnist-8.onnx", std::ios_base::binary);
    model.ParseFromIstream(&file);
    file.close();

    auto batch = 100;

    std::map<std::string, Matrix> map_matrix, map_weight;
    auto graph = model.graph();

    //输入
    for (int i = 0; i < graph.input_size(); ++i)
    {
        auto& input = graph.input(i);
        auto& name = input.name();
        auto& type = input.type();
        auto& shape = type.tensor_type().shape();
        std::vector<int> dim;
        for (int i = 0; i < shape.dim_size(); i++)
        {
            dim.push_back(shape.dim(i).dim_value());
        }
        std::reverse(dim.begin(), dim.end());

        Matrix m(dim);
        if (name.find("Input") != std::string::npos)
        {
            dim.back() = batch;
            m.resize(dim);
            setX(m);
        }
        map_matrix[name] = m;
    }

    //载入权重
    for (int i = 0; i < graph.initializer_size(); ++i)
    {
        auto& init = graph.initializer(i);
        auto& name = init.name();
        auto type = init.data_type();
        auto& mat = map_matrix[name];
        if (type == 1 || type == 6)
        {
            mat.load(init.float_data().data(), init.float_data_size());
        }
        else if (type == 2 || type == 7)
        {
            mat.toCPU();
            for (int i = 0; i < mat.getDataSize(); i++)
            {
                mat.getData(i) = init.int64_data()[i];
            }
        }
        map_weight[name] = mat;
        if (mat.inGPU())
        {
            addWeight(mat);
        }
    }

    //计算图结构
    for (int i = 0; i < graph.node_size(); i++)
    {
        auto& node = graph.node(i);
        auto& node_name = node.name();
        auto& type = node.op_type();

        auto m_in = [&](int i) -> Matrix& { return map_matrix[node.input()[i]]; };
        auto m_out = [&](int i) -> Matrix& { return map_matrix[node.output()[i]]; };

        if (type == "Conv")
        {
            std::vector<int> stride, padding;
            for (auto attr : node.attribute())
            {
                if (attr.name() == "auto_pad")
                {
                    if (attr.s() == "SAME_UPPER")
                    {
                        for (int i = 0; i < m_in(1).getDim().size() - 2; i++)
                        {
                            padding.push_back((m_in(1).getDim()[i] - 1) / 2);
                        }
                    }
                }
            }
            m_out(0) = conv(m_in(0), m_in(1), { 1, 1 }, padding);
        }
        else if (type == "Add")
        {
            auto dim = m_in(1).getDim();
            if (dim.size() != m_in(0).getDim().size())
            {
                dim.push_back(1);
            }
            m_in(1).resize(dim);
            m_out(0) = m_in(0) + m_in(1);
        }
        else if (type == "MaxPool")
        {
            std::vector<int> window, stride, padding;
            for (auto attr : node.attribute())
            {
                if (attr.name() == "kernel_shape")
                {
                    for (int i = 0; i < m_in(0).getDim().size() - 2; i++)
                    {
                        window.push_back(attr.ints()[i]);
                    }
                }
                if (attr.name() == "strides")
                {
                    for (int i = 0; i < m_in(0).getDim().size() - 2; i++)
                    {
                        stride.push_back(attr.ints()[i]);
                    }
                }
            }
            m_out(0) = maxpool(m_in(0), window, stride, padding);
        }
        else if (type == "MatMul")
        {
            m_out(0) = mul(m_in(1), m_in(0));
        }
        else if (type == "Relu")
        {
            m_out(0) = relu(m_in(0));
        }
        else if (type == "Reshape")
        {
            m_out(0) = m_in(0);
            std::vector<int> dim;
            for (int i = 0; i < m_in(1).getDataSize(); i++)
            {
                dim.push_back(m_in(1).getData(i));
            }
            std::reverse(dim.begin(), dim.end());
            if (map_weight.count(node.input()[0]) == 0)    //不是权重，则应该是数据
            {
                dim.back() = batch;
            }
            m_out(0).resize(dim);
        }
        if (i == graph.node_size() - 1)
        {
            setA(softmax_ce(m_out(0)));
        }
    }
    Y_.resize(A_);
    addLoss(crossEntropy(A_, Y_));
}

}    // namespace woco