#pragma once
#include <string>

namespace woco
{

//A buffer to help loading and saving data of matrices
class SaveBuffer
{
public:
    SaveBuffer();

private:
    std::string buffer_;
    int pointer_ = 0;
    std::string filename_;

public:
    int loadFromFile(std::string filename);
    int writeToFile(std::string filename = "");
    int loadFromString(std::string content);

    int load(void* s, size_t size);
    int save(void* s, size_t size);

    size_t size() { return buffer_.size(); }
    void resetPointer() { pointer_ = 0; }
    void setPointer(int p) { pointer_ = p; }
    int getPointer() { return pointer_; }

    std::string& getString();
};

}    // namespace woco