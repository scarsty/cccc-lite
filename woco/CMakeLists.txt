project(woco)
cmake_minimum_required(VERSION 2.6)

file(GLOB SRC_LIST 
    ./*.cpp
    ./*.c
    ../../common/*.cpp
    ../../common/*.c)

set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_FLAGS "-O3 -std=c++17 -pthread -fopenmp -DNDEBUG")

include_directories(/usr/local/cuda/include /usr/local/cuda/include/crt ../../common ../woco-cuda ../include .)
link_directories(/usr/local/cuda/lib64 /usr/local/cuda/lib64/stubs)

set(LINK_LIBRARIES woco-cuda cudnn cudart curand cublas openblas lua5.3)
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^arm" AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64")
    list(APPEND LINK_LIBRARIES nvidia-ml)
endif()

add_library(woco SHARED ${SRC_LIST})
target_link_libraries(woco LINK_PRIVATE ${LINK_LIBRARIES})

install(TARGETS woco LIBRARY DESTINATION lib)
