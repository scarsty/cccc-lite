#include "Test.h"
#include "CudaControl.h"
#include "File.h"
#include "Log.h"
#include "Matrix.h"
#include "MatrixOperator.h"
#include "Net.h"
#include "NetCifa.h"
#include "Timer.h"
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace woco
{

void Test::test2(int mode)
{
    double t_cpu = 0, t_gpu = 0;
    if (mode == 0 || mode == 1)
    {
        CudaControl::checkDevices();
        CudaControl::setGlobalCudaType(DeviceType::GPU);
        CudaControl::select(0);
        Log::LOG("GPU\n");
        t_gpu = testAll();
        CudaControl::destroyAll();
    }
    if (mode == 0 || mode == 2)
    {
        CudaControl::setGlobalCudaType(DeviceType::CPU);
        Log::LOG("CPU\n");
        t_cpu = testAll();
    }
    Log::LOG("Time GPU = %g, Time CPU = %g, CPU slow with %g times.\n", t_gpu, t_cpu, t_cpu / t_gpu);
}

double Test::testAll()
{
    std::vector<double> total =
    {
        testMatrix(),
        //testNet(),
    };
    double t = std::accumulate(total.begin(), total.end(), 0);
    return t;
}

double Test::testMatrix()
{
    Timer t;
    t.start();
    Matrix A(5, 3);
    Matrix B(3, 4);
    A.initRandom();
    B.initRandom();
    A.printAsMatrix();
    B.printAsMatrix();
    Matrix f;
    auto C = A * B;
    C.message("Matrix C:");
    C.printAsMatrix();
    //C.toCPU();
    //C.message("Matrix C:");
    //C.printAsMatrix();
    //C.toGPU();
    //C.message("Matrix C:");
    //C.printAsMatrix();
    C.resize(9, 9, 1, 1);
    C.message("Matrix C:");
    C.printAsMatrix();
    return t.getElapsedTime();
}

double Test::testNet()
{
    Timer t;
    NetCifa net;
    //MatrixOperator::beginMaking(1);
    INIReaderNormal ini;
    ini.loadFile("mnist.ini");
    std::string script = ini.getString("net", "structure");
    printf("%s\n", script.c_str());
    net.setScript(script);
    net.makeStructure();
    return t.getElapsedTime();
}

}    // namespace woco

int main(int argc, char* argv[])
{
    woco::Test t;
    t.test2(0);
    return 0;
}