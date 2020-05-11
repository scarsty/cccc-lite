#include "NetLua.h"
#ifdef _WIN32
#include "lua.hpp"
#else
#include "lua5.3/lua.hpp"
#endif

namespace woco
{

NetLua::LuaWarpper::LuaWarpper()
{
    lua_state_ = luaL_newstate();
}

NetLua::LuaWarpper::~LuaWarpper()
{
    lua_close(lua_state_);
}

NetLua::NetLua()
{
    luaL_openlibs(lua_state_->get());
    lua_pushlightuserdata(lua_state_->get(), this);
    lua_setglobal(lua_state_->get(), "this");
    registerFunctions();
}

NetLua* NetLua::getThis(lua_State* L)
{
    lua_getglobal(L, "this");
    return (NetLua*)lua_touserdata(L, -1);
}

std::string NetLua::registerMatrix(lua_State* L, Matrix& m)
{
    static int count;
    std::string str = "matrix_" + std::to_string(count++);
    getThis(L)->map_matrix_[str] = std::move(m);
    return str;
}

std::string NetLua::registerLoss(lua_State* L, MatrixOperator::Queue& loss)
{
    static int count;
    std::string str = "loss_" + std::to_string(count++);
    getThis(L)->map_loss_[str] = std::move(loss);
    return str;
}

std::vector<int> NetLua::getVector(lua_State* L, int index)
{
    std::vector<int> v;
    if (!lua_isnoneornil(L, index) && lua_istable(L, index))
    {
        int n = luaL_len(L, index);
        for (int i = 0; i < n; i++)
        {
            lua_pushnumber(L, i + 1);
            lua_gettable(L, index);
            v.push_back(lua_tonumber(L, -1));
            lua_pop(L, 1);
        }
    }
    return v;
}

Matrix& NetLua::toMatrix(lua_State* L, int index)
{
    return getThis(L)->map_matrix_[lua_tostring(L, index)];
}

MatrixOperator::Queue& NetLua::toLoss(lua_State* L, int index)
{
    return getThis(L)->map_loss_[lua_tostring(L, index)];
}

void NetLua::pushMatrix(lua_State* L, Matrix& m)
{
    lua_pushstring(L, registerMatrix(L, m).c_str());
}

void NetLua::pushLoss(lua_State* L, MatrixOperator::Queue& loss)
{
    lua_pushstring(L, registerLoss(L, loss).c_str());
}

bool NetLua::isMatrix(lua_State* L, int index)
{
    return lua_isstring(L, index) && std::string(lua_tostring(L, index)).find("matrix_") == 0;
}

bool NetLua::isLoss(lua_State* L, int index)
{
    return lua_isstring(L, index) && std::string(lua_tostring(L, index)).find("loss_") == 0;
}

void NetLua::structure()
{
    //cudnnTensorDescriptor_t t;
    //auto s = sizeof(*t);
    weights_.clear();
    loss_.clear();
    runScript(script_);
    map_matrix_.clear();
    map_loss_.clear();

    for (auto& m : weights_)
    {
        MatrixExtend::fill(m, RANDOM_FILL_XAVIER, m.getChannel(), m.getNumber());
    }
}

int NetLua::runScript(const std::string& script)
{
    //printf("%s\n", script.c_str());
    luaL_loadbuffer(lua_state_->get(), script.c_str(), script.size(), "code");
    int r = lua_pcall(lua_state_->get(), 0, 0, 0);
    if (r)
    {
        printf("\nError: %s\n", lua_tostring(lua_state_->get(), -1));
    }
    return r;
}

int NetLua::registerFunctions()
{
    auto L = lua_state_->get();
    lua_register(L, "Matrix", [](lua_State* L)
        {
            int n = lua_gettop(L);
            std::vector<int> dim;
            for (int i = 0; i < n; i++)
            {
                dim.push_back(lua_tonumber(L, i + 1));
            }
            Matrix m(dim);
            pushMatrix(L, m);
            return 1;
        });
    lua_register(L, "print_message", [](lua_State* L)
        {
            fprintf(stdout, "%s\n", lua_tostring(L, 1));
            toMatrix(L, 1).message();
            return 0;
        });
    lua_register(L, "setXYA", [](lua_State* L)
        {
            getThis(L)->setXYA(toMatrix(L, 1), toMatrix(L, 2), toMatrix(L, 3));
            return 0;
        });
    lua_register(L, "clearWeight", [](lua_State* L)
        {
            getThis(L)->weights_.clear();
            return 0;
        });
    lua_register(L, "addWeight", [](lua_State* L)
        {
            int n = lua_gettop(L);
            for (int i = 0; i < n; i++)
            {
                getThis(L)->weights_.push_back(toMatrix(L, i + 1));
            }
            return 0;
        });
    lua_register(L, "add", [](lua_State* L)
        {
            if (isMatrix(L, 1) && isMatrix(L, 2))
            {
                Matrix m = toMatrix(L, 1) + toMatrix(L, 2);
                pushMatrix(L, m);
                return 1;
            }
            if (isLoss(L, 1) && isLoss(L, 2))
            {
                auto loss = toLoss(L, 1) + toLoss(L, 2);
                pushLoss(L, loss);
                return 1;
            }
            return 0;
        });
    lua_register(L, "mul", [](lua_State* L)
        {
            if (isMatrix(L, 1) && isMatrix(L, 2))
            {
                Matrix m = toMatrix(L, 1) * toMatrix(L, 2);
                pushMatrix(L, m);
                return 1;
            }
            if (lua_isnumber(L, 1) && isLoss(L, 2))
            {
                auto loss = lua_tonumber(L, 1) * toLoss(L, 2);
                pushLoss(L, loss);
                return 1;
            }
            return 0;
        });
    lua_register(L, "conv", [](lua_State* L)
        {
            auto stride = getVector(L, 3);
            auto padding = getVector(L, 4);
            Matrix m = conv(toMatrix(L, 1), toMatrix(L, 2), stride, padding);
            pushMatrix(L, m);
            return 1;
        });
    lua_register(L, "maxpool", [](lua_State* L)
        {
            auto window = getVector(L, 2);
            auto stride = getVector(L, 3);
            auto padding = getVector(L, 4);
            Matrix m = maxpool(toMatrix(L, 1), window, stride, padding);
            pushMatrix(L, m);
            return 1;
        });
    lua_register(L, "getRow", [](lua_State* L)
        {
            lua_pushnumber(L, toMatrix(L, 1).getRow());
            return 1;
        });

#define REGISTER(func) \
    lua_register(L, #func, [](lua_State* L) { \
        Matrix m = func(toMatrix(L, 1)); \
        pushMatrix(L, m); \
        return 1; \
    })

    REGISTER(relu);
    REGISTER(sigmoid);
    REGISTER(softmax);
    REGISTER(softmax_ce);

    lua_register(L, "addLoss", [](lua_State* L)
        {
            int n = lua_gettop(L);
            for (int i = 0; i < n; i++)
            {
                getThis(L)->loss_ = getThis(L)->loss_ + toLoss(L, i + 1);
            }
            return 0;
        });
    lua_register(L, "crossEntropy", [](lua_State* L)
        {
            auto q = crossEntropy(toMatrix(L, 1), toMatrix(L, 2));
            pushLoss(L, q);
            return 1;
        });
    lua_register(L, "L2", [](lua_State* L)
        {
            auto q = L2(toMatrix(L, 1));
            pushLoss(L, q);
            return 1;
        });

    runScript(R"!!(getmetatable('').__add = function(a,b) return add(a,b) end)!!");
    runScript(R"!!(getmetatable('').__mul = function(a,b) return mul(a,b) end)!!");
    return 0;
}

}    // namespace woco