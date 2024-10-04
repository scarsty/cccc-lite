#include "DataPreparerTxt.h"
#include "VectorMath.h"
#include "filefunc.h"

namespace cccc
{

DataPreparerTxt::DataPreparerTxt()
{
}

DataPreparerTxt::~DataPreparerTxt()
{
}

void DataPreparerTxt::init2()
{
    input_ = VectorMath::multiply(dim0_, dim0_.size() - 1);
    output_ = VectorMath::multiply(dim1_, dim1_.size() - 1);
    std::string filename = option_->getString(section_, "file", "file.txt");
    content_ = filefunc::readFileToString(filename);
}

void DataPreparerTxt::fillData0()
{
    rand_.set_seed();

    for (int index = 0; index < X.getNumber(); index++)
    {
        int r = rand_.rand() * (content_.size() / 2 - input_ - output_);
        for (int i = 0; i < input_; i++)
        {
            X.setData(i, index, getContent(r + i));
        }
        for (int i = 0; i < output_; i++)
        {
            Y.setData(i, index, getContent(r + i + input_));
        }
    }
}

float DataPreparerTxt::getContent(int i)
{
    auto p = (uint16_t*)(&content_[i * 2]);
    return (*p) / 65536.0;
}

}    // namespace cccc