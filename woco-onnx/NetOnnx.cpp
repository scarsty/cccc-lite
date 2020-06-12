#include "NetOnnx.h"
#include "onnx.proto3.pb.h"

namespace woco
{

void NetOnnx::structure()
{
    //NetCifa::structure();
    //return;

    onnx::ModelProto model;
    //std::ifstream file("mnist-8.onnx", std::ios_base::binary);
    model.ParseFromString(convert::readStringFromFile("test.onnx"));
    //file.close();

    auto batch = 1;

    std::map<std::string, Matrix> map_matrix, map_weight;
    auto graph = model.graph();

    auto get_matrix = [&](const onnx::ValueInfoProto& vip) -> Matrix&
    {
        auto& name = vip.name();
        auto& type = vip.type();
        auto& shape = type.tensor_type().shape();
        std::vector<int> dim;
        for (int i = 0; i < shape.dim_size(); i++)
        {
            dim.push_back(shape.dim(i).dim_value());
        }
        std::reverse(dim.begin(), dim.end());
        if (dim.size() == 3)
        {
            dim.push_back(1);
        }
        Matrix m(dim);
        map_matrix[name] = m;
        return map_matrix[name];
    };

    for (int i = 0; i < graph.input_size(); ++i)
    {
        get_matrix(graph.input(i));
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

    for (auto& p : map_matrix)
    {
        if (map_weight.count(p.first) == 0)
        {
            auto& m = map_matrix[p.first];
            auto dim = m.getDim();
            dim.back() = batch;
            m.resize(dim);
            setX(m);
        }
    }
    for (int i = 0; i < graph.value_info_size(); ++i)
    {
        get_matrix(graph.value_info(i));
    }
    for (int i = 0; i < graph.output_size(); ++i)
    {
        A_ = get_matrix(graph.output(i));
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
                if (attr.name() == "strides")
                {
                    for (int i = 0; i < m_in(0).getDim().size() - 2; i++)
                    {
                        stride.push_back(attr.ints()[i]);
                    }
                }
                if (attr.name() == "pads")
                {
                    for (int i = 0; i < m_in(0).getDim().size() - 2; i++)
                    {
                        padding.push_back(attr.ints()[i]);
                    }
                }
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
            getDefaultStridePadding(MatrixOpType::CONV, m_in(1).getDim(), stride, padding);
            conv(m_in(0), m_in(1), m_out(0), stride, padding);
        }
        else if (type == "Add")
        {
            add(m_in(0), m_in(1), m_out(0));
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
            getDefaultStridePadding(MatrixOpType::POOL, window, stride, padding);
            pool(m_in(0), m_out(0), POOLING_MAX, window, stride, padding);
        }
        else if (type == "MatMul")
        {
            mul(m_in(1), m_in(0), m_out(0));
        }
        else if (type == "Relu")
        {
            woco::active(m_in(0), m_out(0), ACTIVE_FUNCTION_RELU);
        }
        else if (type == "Softmax")
        {
            woco::active(m_in(0), m_out(0), ACTIVE_FUNCTION_SOFTMAX);
        }
        else if (type == "Reshape")
        {
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
            reshape(m_in(0), m_out(0), dim);
        }
    }
    Y_.resize(A_);
    addLoss(crossEntropy(A_, Y_));
}

void NetOnnx::save(const std::string& filename)
{
    auto graph = new onnx::GraphProto();

    auto matrix_string = [](const Matrix& m)
    {
        std::string s;
        for (auto i : m.getDim())
        {
            s += "_" + std::to_string(i);
        }
        return "Matrix_" + std::to_string(int64_t(m.getDataPointer())) + s;
    };

    auto register_matrix = [&matrix_string](const Matrix& m, onnx::ValueInfoProto* vip, int data_type = 1)
    {
        vip->set_name(matrix_string(m));
        auto type = new onnx::TypeProto();
        auto tensor_type = new onnx::TypeProto_Tensor();
        auto shape = new onnx::TensorShapeProto();
        auto dim = m.getDim();
        std::reverse(dim.begin(), dim.end());
        for (auto i : dim)
        {
            auto a = shape->add_dim();
            a->set_dim_value(i);
        }
        tensor_type->set_elem_type(data_type);
        tensor_type->set_allocated_shape(shape);
        type->set_allocated_tensor_type(tensor_type);
        vip->set_allocated_type(type);
    };
    auto init_matrix = [&matrix_string](const Matrix& m, onnx::TensorProto* tp, int data_type = 1)
    {
        tp->set_name(matrix_string(m));
        tp->set_data_type(data_type);

        auto dim = m.getDim();
        std::reverse(dim.begin(), dim.end());
        for (auto i : dim)
        {
            tp->add_dims(i);
        }
        auto m1 = m.clone(DeviceType::CPU);
        if (data_type == 1 || data_type == 6)
        {
            for (int i = 0; i < m1.getDataSize(); i++)
            {
                tp->add_float_data(m1.getData(i));
            }
        }
        else if (data_type == 2 || data_type == 7)
        {
            for (int i = 0; i < m1.getDataSize(); i++)
            {
                tp->add_int64_data(m1.getData(i));
            }
        }
    };

    register_matrix(X_, graph->add_input());
    register_matrix(A_, graph->add_output());

    for (auto& m : weights_)
    {
        register_matrix(m, graph->add_input());
        init_matrix(m, graph->add_initializer());
    }
    int index = 0;
    for (auto& op : op_queue_)
    {
        auto node = graph->add_node();
        node->set_name("Op_" + std::to_string(index++));

        if (op.getType() == MatrixOpType::MUL)
        {
            *node->add_input() = matrix_string(op.getMatrixIn()[1]);
            *node->add_input() = matrix_string(op.getMatrixIn()[0]);
        }
        else
        {
            for (const auto& m : op.getMatrixIn())
            {
                *node->add_input() = matrix_string(m);
            }
        }
        for (const auto& m : op.getMatrixOut())
        {
            *node->add_output() = matrix_string(m);
            register_matrix(m, graph->add_value_info());
        }

        switch (op.getType())
        {
        case MatrixOpType::ADD:
            node->set_op_type("Add");
            break;
        case MatrixOpType::MUL:
            node->set_op_type("MatMul");
            break;
        case MatrixOpType::ADD_BIAS:
            node->set_op_type("Add");
            break;
        case MatrixOpType::ACTIVE:
            switch (ActiveFunctionType(op.getPataInt().back()))
            {
            case ACTIVE_FUNCTION_SIGMOID:
            case ACTIVE_FUNCTION_SIGMOID_CE:
                node->set_op_type("Sigmoid");
                break;
            case ACTIVE_FUNCTION_RELU:
                node->set_op_type("Relu");
                break;
            case ACTIVE_FUNCTION_TANH:
                node->set_op_type("Tanh");
                break;
            case ACTIVE_FUNCTION_SOFTMAX:
            case ACTIVE_FUNCTION_SOFTMAX_CE:
                node->set_op_type("Softmax");
                break;
            }
            break;
        case MatrixOpType::POOL:
            node->set_op_type("MaxPool");
            {
                auto a = node->add_attribute();
                a->set_name("auto_pad");
                a->set_s("NOTSET");
                auto a0 = node->add_attribute();
                a0->set_name("kernel_shape");
                for (auto i : op.getPataInt2()[0])
                {
                    a0->add_ints(i);
                }
                auto a1 = node->add_attribute();
                a1->set_name("strides");
                for (auto i : op.getPataInt2()[1])
                {
                    a1->add_ints(i);
                }
            }
            break;
        case MatrixOpType::CONV:
            node->set_op_type("Conv");
            {
                auto a = node->add_attribute();
                a->set_name("auto_pad");
                a->set_s("NOTSET");
                auto a0 = node->add_attribute();
                a0->set_name("kernel_shape");
                auto dim = op.getMatrixIn()[1].getDim();
                dim.pop_back();
                dim.pop_back();
                for (auto i : dim)
                {
                    a0->add_ints(i);
                }
                auto a1 = node->add_attribute();
                a1->set_name("strides");
                for (auto i : op.getPataInt2()[0])
                {
                    a1->add_ints(i);
                }
                auto a2 = node->add_attribute();
                a2->set_name("pads");
                for (auto i : op.getPataInt2()[1])
                {
                    a2->add_ints(i);
                    a2->add_ints(i);
                }
            }
            break;
        case MatrixOpType::RESHAPE:
            node->set_op_type("Reshape");
            {
                auto dim = op.getMatrixOut()[0].getDim();
                std::reverse(dim.begin(), dim.end());
                Matrix m({ int(dim.size()) }, DeviceType::CPU);
                for (int i = 0; i < dim.size(); i++)
                {
                    m.getData(i) = dim[i];
                }
                init_matrix(m, graph->add_initializer(), 7);
                *node->add_input() = matrix_string(m);
                register_matrix(m, graph->add_input(), 7);
            }
        }
    }

    onnx::ModelProto model;
    model.set_allocated_graph(graph);

    model.set_ir_version(3);
    model.set_producer_name("woco");
    model.set_producer_version("0");
    model.set_model_version(1);
    std::string str;

    //graph.
    model.SerializePartialToString(&str);
    convert::writeStringToFile(str, "test.onnx");
    std::cout << str << std::endl;
}

}    // namespace woco