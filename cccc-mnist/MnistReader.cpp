#include "MnistReader.h"
#include "Log.h"
#include "filefunc.h"

namespace cccc
{

MnistReader::MnistReader()
{
}

void MnistReader::getDataSize(const std::string& file_image, int* w, int* h, int* n)
{
    auto content = filefunc::readFile(file_image, 16);
    reverse(content.data() + 4, 4);
    reverse(content.data() + 8, 4);
    reverse(content.data() + 12, 4);
    if (n)
    {
        *n = *(int*)(content.data() + 4);
    }
    if (w)
    {
        *w = *(int*)(content.data() + 8);
    }
    if (h)
    {
        *h = *(int*)(content.data() + 12);
    }
}

void MnistReader::readLabelFile(const std::string& filename, real* y_data)
{
    int s = 10;
    auto content = filefunc::readFile(filename);
    reverse(content.data() + 4, 4);
    int n = *(int*)(content.data() + 4);
    memset(y_data, 0, sizeof(real) * n * s);
    for (int i = 0; i < n; i++)
    {
        int pos = *(content.data() + 8 + i);
        y_data[i * s + pos % s] = 1;
    }
}

void MnistReader::readImageFile(const std::string& filename, real* x_data)
{
    auto content = filefunc::readFile(filename);
    reverse(content.data() + 4, 4);
    reverse(content.data() + 8, 4);
    reverse(content.data() + 12, 4);
    int n = *(int*)(content.data() + 4);
    int w = *(int*)(content.data() + 8);
    int h = *(int*)(content.data() + 12);
    int size = n * w * h;
    memset(x_data, 0, sizeof(real) * size);
    for (int i = 0; i < size; i++)
    {
        auto v = *(uint8_t*)(content.data() + 16 + i);
        x_data[i] = real(v / 255.0);
    }
}

void MnistReader::readData(const std::string& file_label, const std::string& file_image, Matrix& X, Matrix& Y)
{
    int w, h, n;
    getDataSize(file_image, &w, &h, &n);
    X.resize(w, h, 1, n);
    Y.resize(label_, n);
    //train.createA();
    readImageFile(file_image, X.getDataPointer());
    readLabelFile(file_label, Y.getDataPointer());
}

void MnistReader::reverse(char* c, int n)
{
    for (int i = 0; i < n / 2; i++)
    {
        auto& a = *(c + i);
        auto& b = *(c + n - 1 - i);
        auto t = b;
        b = a;
        a = t;
    }
}
void MnistReader::load(Matrix& X, Matrix& Y, std::string path /*= "mnist"*/, int data_type /*= 1*/)
{
    if (path.back() != '/' || path.back() != '\\')
    {
        path += '/';
    }
    std::string label;
    std::string image;
    if (data_type == 1)
    {
        label = path + "train-labels.idx1-ubyte";
        image = path + "train-images.idx3-ubyte";
        LOG("Loading MNIST train data... ");
    }
    else if (data_type == 2)
    {
        label = path + "t10k-labels.idx1-ubyte";
        image = path + "t10k-images.idx3-ubyte";
        LOG("Loading MNIST test data... ");
    }
    else
    {
        LOG("Please check MNIST type: 1 - train set (60000), 2 - test set (10000)\n");
        return;
    }
    readData(label, image, X, Y);
    LOG("done\n");
}

}    // namespace cccc