#pragma once
#include "Net.h"

class lua_State;

namespace woco
{

class DLL_EXPORT NetLua : public Net
{
private:
    struct LuaWarpper
    {
        lua_State* lua_state_ = nullptr;
        LuaWarpper();
        ~LuaWarpper();
        LuaWarpper(const LuaWarpper&) = delete;
        LuaWarpper& operator=(const LuaWarpper) = delete;
        lua_State* get() { return lua_state_; }
    };

    std::shared_ptr<LuaWarpper> lua_state_ = std::make_shared<LuaWarpper>();
    std::map<std::string, Matrix> map_matrix_;
    std::map<std::string, MatrixOperator::Queue> map_loss_;
    std::string script_;

    static NetLua* getThis(lua_State* L);

    static std::string registerMatrix(lua_State* L, Matrix& m);
    static std::string registerLoss(lua_State* L, MatrixOperator::Queue& loss);
    static std::vector<int> getVector(lua_State* L, int index);

    static Matrix& toMatrix(lua_State* L, int index);
    static MatrixOperator::Queue& toLoss(lua_State* L, int index);

    static void pushMatrix(lua_State* L, Matrix& m);
    static void pushLoss(lua_State* L, MatrixOperator::Queue& loss);

    static bool isMatrix(lua_State* L, int index);
    static bool isLoss(lua_State* L, int index);

public:
    NetLua();
    virtual void structure() override;
    void setScript(const std::string& script) { script_ = script; }

    int runScript(const std::string& script);
    int registerFunctions();
};

}    // namespace woco