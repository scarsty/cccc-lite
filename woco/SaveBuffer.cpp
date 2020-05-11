#include "SaveBuffer.h"
#include "convert.h"
#include <cstring>

namespace woco
{

SaveBuffer::SaveBuffer()
{
}

int SaveBuffer::loadFromFile(std::string filename)
{
    filename_ = filename;
    buffer_ = convert::readStringFromFile(filename);
    return buffer_.size();
}

int SaveBuffer::writeToFile(std::string filename)
{
    if (filename == "")
    {
        filename = filename_;
    }
    return convert::writeStringToFile(buffer_, filename);
}

int SaveBuffer::loadFromString(std::string content)
{
    buffer_ = content;
    return buffer_.size();
}

int SaveBuffer::load(void* s, size_t size)
{
    if (buffer_.size() >= pointer_ + size)
    {
        memcpy(s, buffer_.data() + pointer_, size);
        pointer_ += size;
    }
    else
    {
        pointer_ += size;
        size = 0;
    }
    return size;
}

int SaveBuffer::save(void* s, size_t size)
{
    if (buffer_.size() < pointer_ + size)
    {
        buffer_.resize(pointer_ + size);
    }
    memcpy((void*)(buffer_.data() + pointer_), s, size);
    pointer_ += size;
    return size;
}

std::string& SaveBuffer::getString()
{
    return buffer_;
}

}    // namespace woco