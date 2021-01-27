#include "MnistReader.h"
#include "File.h"

namespace will
{

MnistReader::MnistReader()
{
}

void MnistReader::getDataSize(const std::string& file_image, int* w, int* h, int* n)
{
    auto content = File::readFile(file_image, 16);
    File::reverse(content.data() + 4, 4);
    File::reverse(content.data() + 8, 4);
    File::reverse(content.data() + 12, 4);
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
    auto content = File::readFile(filename);
    File::reverse(content.data() + 4, 4);
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
    auto content = File::readFile(filename);
    File::reverse(content.data() + 4, 4);
    File::reverse(content.data() + 8, 4);
    File::reverse(content.data() + 12, 4);
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
        fprintf(stdout, "Loading MNIST train data... ");
    }
    else if (data_type == 2)
    {
        label = path + "t10k-labels.idx1-ubyte";
        image = path + "t10k-images.idx3-ubyte";
        fprintf(stdout, "Loading MNIST test data... ");
    }
    else
    {
        fprintf(stdout, "Please check MNIST type: 1 - train set (60000), 2 - test set (10000)\n");
        return;
    }
    readData(label, image, X, Y);
    fprintf(stdout, "done\n");
}

}    // namespace will