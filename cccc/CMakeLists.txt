project(cccc)
cmake_minimum_required(VERSION 2.6)

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

include_directories(/usr/local/cuda/include /usr/local/cuda/include/crt ../../mlcc ../include . ./layer ../cccc-cuda ../cccc-hip)
link_directories(/usr/local/cuda/lib64)

set(LINK_LIBRARIES cudnn cudart cublas openblas)
if(NOT CCCC_CPU)
    include_directories(../cccc-cuda)
    list(APPEND LINK_LIBRARIES cccc-cuda)
endif()

add_library(cccc SHARED ${SRC_LIST})
target_link_libraries(cccc LINK_PRIVATE ${LINK_LIBRARIES})

install(TARGETS cccc LIBRARY DESTINATION lib)
