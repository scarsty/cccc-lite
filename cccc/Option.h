#pragma once
#include "ConsoleControl.h"
#include "Log.h"
#include "strfunc.h"
#include "types.h"
#include <functional>
#include <string>
#include <typeinfo>
#define VIRTUAL_GET_STRING
#include "INIReader.h"

namespace cccc
{

//该类用于读取配置文件，并转换其中的字串设置为枚举
//注意实数只获取双精度数，如果是单精度模式会包含隐式转换
//获取整数的时候，先获取双精度数并强制转换

class Option : public INIReaderNoUnderline
{
public:
    Option();
    Option(const std::string& filename);

public:
    std::string dealString(std::string str, bool allow_path);

    //以下为枚举值的处理
private:
    struct CompareNoUnderline
    {
        bool operator()(const std::string& l, const std::string& r) const
        {
            auto l1 = l;
            auto r1 = r;
            auto replaceAllString = [](std::string& s, const std::string& oldstring, const std::string& newstring)
            {
                int pos = s.find(oldstring);
                while (pos >= 0)
                {
                    s.erase(pos, oldstring.length());
                    s.insert(pos, newstring);
                    pos = s.find(oldstring, pos + newstring.length());
                }
            };
            replaceAllString(l1, "_", "");
            replaceAllString(r1, "_", "");
            std::transform(l1.begin(), l1.end(), l1.begin(), ::tolower);
            std::transform(r1.begin(), r1.end(), r1.begin(), ::tolower);
            return l1 < r1;
        }
    };
    std::map<std::string, std::map<std::string, int, CompareNoUnderline>> enum_map_;
    std::map<std::string, std::map<int, std::string>> enum_map_reverse_;
    //注册枚举值
    template <typename T>
    void registerEnum(std::vector<std::pair<std::string, T>> members)
    {
        for (auto m : members)
        {
            enum_map_[typeid(T).name()][m.first] = int(m.second);
            //反查map只保留第一个，注册时需注意顺序
            if (enum_map_reverse_[typeid(T).name()].count(int(m.second)) == 0)
            {
                enum_map_reverse_[typeid(T).name()][int(m.second)] = m.first;
            }
        }
    }
    void initEnums();

    mutable std::map<std::string, std::map<std::string, int>> outputed_keys_;

private:
    template <typename T>
    void outputValue(const std::string& section, const std::string& key, const T& value, const T& default_value) const
    {
        int count = outputed_keys_[section][key]++;
        if (output_ == 0
            || (output_ == 2 && value == default_value))
        {
            return;
        }
        if (count == 0)
        {
            ConsoleControl::setColor(CONSOLE_COLOR_GREEN);
            LOG("  [\"{}\"][\"{}\"] = {}\n", section, key, value);
            ConsoleControl::setColor(CONSOLE_COLOR_NONE);
        }
    }
    int output_ = 1;    //0表示不输出，1表示输出，2表示若与默认值不同则输出

public:
    void setOutput(int output)
    {
        output_ = output;
    }
    std::string getString(const std::string& section, const std::string& key, const std::string& default_value = "") const
    {
        auto value = INIReaderNoUnderline::getString(section, key, default_value);
        outputValue(section, key, value.substr(0, value.find_first_of(";\n")), default_value);
        return value;
    }
    int getInt(const std::string& section, const std::string& key, int default_value = 0) const
    {
        auto value = INIReaderNoUnderline::getInt(section, key, default_value);
        outputValue(section, key, value, default_value);
        return value;
    }
    float getReal(const std::string& section, const std::string& key, float default_value = 0) const
    {
        float value = INIReaderNoUnderline::getReal(section, key, default_value);
        outputValue(section, key, value, default_value);
        return value;
    }
    template <typename T>
    std::vector<T> getVector(const std::string& section, const std::string& key, const std::string& split_chars = ",", const std::vector<T>& default_v = {}) const
    {
        auto value = INIReaderNoUnderline::getVector<T>(section, key, split_chars, default_v);
        outputValue(section, key, value, default_v);
        return value;
    }
    //将字串转为枚举值
    template <typename T>
    T getEnumFromString(const std::string& value_str)
    {
        return T(enum_map_[typeid(T).name()][value_str]);
    }
    //从配置中直接读出枚举值
    //按照C++推荐，最后的参数默认值应为T{}，但swig不能正确识别
    template <typename T>
    T getEnum(const std::string& section, const std::string& key, T default_value = T(0))
    {
        std::string value_str = INIReaderNoUnderline::getString(section, key);
        T v = default_value;
        if (enum_map_[typeid(T).name()].count(value_str) > 0)
        {
            v = T(enum_map_[typeid(T).name()][value_str]);
        }
        else
        {
            if (!value_str.empty())
            {
                LOG("Warning: undefined value \"{}\" for {}, set to {}!\n", value_str, key, getStringFromEnum(T(0)));
            }
        }
        outputValue(section, key, getStringFromEnum(v), getStringFromEnum(default_value));
        return v;
    }

    //反查枚举值为字串
    template <typename T>
    std::string getStringFromEnum(T e)
    {
        return enum_map_reverse_[typeid(T).name()][int(e)];
    }

public:
    //去掉下划线，使输出略为美观
    static std::string removeEndUnderline(const std::string& str)
    {
        if (!str.empty() && str.back() == '_')
        {
            return str.substr(0, str.size() - 1);
        }
        return str;
    }
};

//以下宏用于简化参数的读取，不可用于其他，必须预先定义option和section
#define NAME_STR(a) (Option::removeEndUnderline(#a).c_str())
#define OPTION_GET_INT(a) \
    do { \
        a = option->getInt(section, NAME_STR(a), a); \
    } while (0)
#define OPTION_GET_REAL(a) \
    do { \
        a = option->getReal(section, NAME_STR(a), a); \
    } while (0)
#define OPTION_GET_STRING(a) \
    do { \
        a = option->getString(section, NAME_STR(a), a); \
    } while (0)
#define OPTION_GET_NUMVECTOR(v, size, fill) \
    do { \
        v.resize(size, fill); \
        v = option->getVector<double>(section, NAME_STR(v), ",", v); \
    } while (0)
#define OPTION_GET_STRINGVECTOR(v) \
    do { \
        v = option->getVector<std::string>(section, NAME_STR(v), ",", v); \
    } while (0)

}    // namespace cccc