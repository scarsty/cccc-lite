#include "NetOnnx.h"
#include "onnx.proto3.pb.h"
#include <fstream>

namespace woco
{

void NetOnnx::structure()
{
    onnx::ModelProto model;
    {

        std::ifstream file("mnist-8.onnx", std::ios_base::binary);
        model.ParseFromIstream(&file);
        file.close();
        std::cout << model.graph().input().size() << "\n";
    }
    std::map<std::string, Matrix> map_matrix;

    onnx::GraphProto graph = model.graph();    //访问graph结构
    int num = graph.node_size();               //节点个数

    int input_size = graph.input_size();    //网络输入个数，input以及各层常量输入
    for (int i = 0; i < input_size; ++i)
    {
        const std::string name = graph.input(i).name();
        onnx::TypeProto type = graph.input(i).type();
        onnx::TensorShapeProto shape = type.tensor_type().shape();    //输入维度
        std::vector<int> size;
        std::cout << name << ": ";
        for (int i = 0; i < shape.dim_size(); i++)
        {
            std::cout << shape.dim(i).dim_value() << " ";
            size.push_back(shape.dim(i).dim_value());
        }
        std::reverse(size.begin(), size.end());
        map_matrix[name] = Matrix(size);
        std::cout << std::endl;
    }
    std::cout << std::endl;
    int output_size = graph.output_size();    //网络output个数
    for (int i = 0; i < output_size; ++i)
    {
        const std::string name = graph.input(i).name();
        onnx::TypeProto type = graph.input(i).type();
        onnx::TensorShapeProto shape = type.tensor_type().shape();    //输入维度
        std::vector<int> size;
        //std::cout << name << ": ";
        for (int i = 0; i < shape.dim_size(); i++)
        {
            //std::cout << shape.dim(i).dim_value() << " ";
            size.push_back(shape.dim(i).dim_value());
        }
        std::reverse(size.begin(), size.end());
        map_matrix[name] = Matrix(size);
        setX(map_matrix[name]);
        // std::cout << std::endl;
    }

    int init_size = graph.initializer_size();    //网络输入个数，input以及各层常量输入
    for (int i = 0; i < init_size; ++i)
    {
        const std::string name = graph.initializer(i).name();
        auto init = graph.initializer(i);
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
        std::vector<int> size;
    }

    std::vector<MatrixOperator> ops;

    for (int i = 0; i < num; i++)    //遍历每个node结构
    {
        const onnx::NodeProto node = graph.node(i);
        std::string node_name = node.name();
        std::cout << "cur node name:" << node_name << std::endl;
        const ::google::protobuf::RepeatedPtrField<::onnx::AttributeProto> attr = node.attribute();    //每个node结构的参数信息
        std::string type = node.op_type();
        int in_size = node.input_size();
        //auto in = node.
        int out_size = node.output_size();
        std::cout << type << ": " << in_size << " " << output_size << ";\n";

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
            m_out(0) = add(m_in(0), m_in(1));
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
            auto m = m_in(1).clone(DeviceType::CPU);
            auto dim = m.getDim();
            std::reverse(dim.begin(), dim.end());
            m.resize(dim);
            m.transpose();
            Matrix::copyData(m, m_in(1));
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
            { dim.push_back(m_in(1).getData(i)); }
            std::reverse(dim.begin(), dim.end());
            m_out(0).resize(dim);
        }
        auto a = node.input();
        for (auto a1 : a)
        {
            std::cout << a1 << ",";
        }
        std::cout << "\n";
        a = node.output();
        for (auto a1 : a)
        {
            std::cout << a1 << ",";
        }
        std::cout << "\n\n";
        if (i == num - 1)
        {
            setA(m_out(0));
        }
    }
}

}    // namespace woco