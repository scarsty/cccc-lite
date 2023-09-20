project(cccc)
cmake_minimum_required(VERSION 2.6)

set(CUDA_INCLUDE_PATH "/usr/local/cuda/include")#Linux下编译需要自行配置cuda以及openblas安装路径
set(CUDA_LIB_PATH "/usr/local/cuda/lib64")
set(OpenBLAS_LIB_PATH "/opt/OpenBLAS/lib")

file(GLOB SRC_LIST
    ./*.cpp
    ./layer/*.cpp
    ./*.c
    ../../mlcc/*.cpp
    ../../mlcc/*.c)

set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_FLAGS "-O3 -std=c++17 -pthread -fopenmp -DNDEBUG")

add_definitions(-DENABLE_CUDA=1 -DENABLE_HIP=0)

include_directories(${CUDA_INCLUDE_PATH} ${CUDA_INCLUDE_PATH}/crt ../../mlcc ../include . ./layer ../cccc-cuda ../cccc-hip)
link_directories(${CUDA_LIB_PATH} ${OpenBLAS_LIB_PATH})

set(LINK_LIBRARIES cudnn cudart cublas openblas)
if(NOT CCCC_CPU)
    include_directories(../cccc-cuda)
    list(APPEND LINK_LIBRARIES cccc-cuda)
endif()

add_library(cccc SHARED ${SRC_LIST})
target_link_libraries(cccc LINK_PRIVATE ${LINK_LIBRARIES})

install(TARGETS cccc LIBRARY DESTINATION lib)
