#pragma once
#include "INIReader.h"
#include "convert.h"
#include "types.h"
#include <functional>
#include <string>
#include <typeinfo>

namespace cccc
{

//该类用于读取配置文件，并转换其中的字串设置为枚举
//注意实数只获取双精度数，如果是单精度模式会包含隐式转换
//获取整数的时候，先获取双精度数并强制转换

class Option
{
public:
    Option();
    Option(const std::string& filename);
    ~Option();

private:
    static const char default_section_[];
    INIReader<CompareDefaultValue<default_section_>, CompareNoUnderline> ini_reader_;

public:
    //载入ini文件
    void loadIniFile(const std::string& filename);
    void loadIniString(const std::string& content);

    //从指定块读取
    int getInt(const std::string& section, const std::string& key, int default_value = 0);
    double getReal(const std::string& section, const std::string& key, double default_value = 0.0);
    std::string getString(const std::string& section, const std::string& key, const std::string& default_value = "");
    template <typename T>
    std::vector<T> getVector(const std::string& section, const std::string& key, const std::vector<T>& default_value = {})
    {
        std::vector<T> v;
        auto str = getString(section, key);
        convert::findNumbers(str, v);
        return v;
    }

    bool hasSection(const std::string& section) { return ini_reader_.hasSection(section); }
    bool hasOption(const std::string& section, const std::string key) { return ini_reader_.hasKey(section, key); }

    std::vector<std::string> getAllSections() { return ini_reader_.getAllSections(); }
    std::vector<std::string> getAllOptions(const std::string& section) { return ini_reader_.getAllKeys(section); }

    void setOption(const std::string& section, const std::string& key, const std::string& value)
    {
        ini_reader_.setKey(section, key, value);
    }
    void setOptions(const std::string& section, const std::string& pairs);
    void setOptions(const std::string& section, const std::vector<std::string>& pairs);

    void print();

    //以下为枚举值的处理
private:
    std::map<std::string, std::map<std::string, int, CompareNoUnderline>> enum_map_;
    std::map<std::string, std::map<int, std::string>> enum_map_reverse_;
    //注册枚举值
    template <typename T>
    void registerEnum(std::vector<std::pair<std::string, T>> members)
    {
        for (auto m : members)
        {
            enum_map_[typeid(T).name()][m.first] = m.second;
            //反查map只保留第一个，注册时需注意顺序
            if (enum_map_reverse_[typeid(T).name()].count(m.second) == 0)
            {
                enum_map_reverse_[typeid(T).name()][m.second] = m.first;
            }
        }
    }
    void initEnums();

public:
    //将字串转为枚举值
    template <typename T>
    T transEnum(const std::string& value_str)
    {
        return T(enum_map_[typeid(T).name()][value_str]);
    }
    //从配置中直接读出枚举值
    //按照C++推荐，最后的参数默认值应为T{}，但swig不能正确识别
    template <typename T>
    T getEnum(const std::string& section, const std::string& key, T default_value = T(0))
    {
        std::string value_str = getString(section, key);
        if (enum_map_[typeid(T).name()].count(value_str) > 0)
        {
            return T(enum_map_[typeid(T).name()][value_str]);
        }
        else
        {
            if (!value_str.empty())
            {
                fprintf(stderr, "Warning: undefined value \"%s\" for %s, set to %s!\n", value_str.c_str(), key.c_str(), getStringFromEnum(T(0)).c_str());
            }
            return default_value;
        }
    }

    //反查枚举值为字串
    template <typename T>
    std::string getStringFromEnum(T e)
    {
        return enum_map_reverse_[typeid(T).name()][e];
    }

public:
    //先读公共块，再读指定块
#define GET_VALUE2(type, name) \
    type name##2(const std::string& s, const std::string& k, type v) { return name(s, k, name("", k, v)); }

    GET_VALUE2(int, getInt)
    GET_VALUE2(real, getReal)
    GET_VALUE2(std::string, getString)

    template <typename T>
    GET_VALUE2(T, getEnum)

#undef GET_VALUE2
};

//#define OPTION_GET_VALUE_INT(op, v, default_v) v = op->getInt("", #v, default_v)

}    // namespace cccc