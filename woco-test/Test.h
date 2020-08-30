#pragma once

namespace woco
{

class Test
{
public:
    void test2(int mode = 0);

private:
    //double返回值都是耗时
    //统计的是纯计算耗时，故不应一起计算
    double testAll();
    double testMatrix();
    double testNet();
    double testActive();
    double testConv();
};

}    // namespace woco
