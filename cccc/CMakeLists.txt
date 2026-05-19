project(cccc)
cmake_minimum_required(VERSION 2.6)

file(GLOB SRC_LIST
    ./*.cpp
    ./layer/*.cpp
    ./*.c
    ../../mlcc/Cifa.cpp
    ../../mlcc/filefunc.cpp
    ../../mlcc/strfunc.cpp
    )

set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_FLAGS "-O3 -std=c++23 -pthread -fopenmp -DNDEBUG")

include_directories(/usr/include /usr/local/include /usr/local/cuda/include /usr/local/cuda/include/crt ../../mlcc ../include . ./layer ../cccc-cuda ../cccc-hip)
link_directories(/usr/local/cuda/lib64 /usr/lib64)

set(LINK_LIBRARIES  openblas)
if(CCCC_CUDA)
    list(APPEND LINK_LIBRARIES cccc-cuda cudnn cudart cublas)
endif()
if(CCCC_HIP)
    list(APPEND LINK_LIBRARIES cccc-hip MIOpen rocblas)
endif()

add_library(cccc SHARED ${SRC_LIST})
target_link_libraries(cccc LINK_PRIVATE ${LINK_LIBRARIES})

install(TARGETS cccc LIBRARY DESTINATION lib)
