project(woco-test)
cmake_minimum_required(VERSION 2.6)

aux_source_directory(./ SRC_LIST)

set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_FLAGS "-O3 -std=c++17 -pthread -fopenmp -DNDEBUG")

include_directories(/usr/local/cuda/include /usr/local/cuda/include/crt ../../common ../woco-cuda ../include ../woco)
link_directories(/usr/local/cuda/lib64 /usr/local/cuda/lib64/stubs)

add_executable(woco-test ${SRC_LIST})
target_link_libraries(woco-test woco cudart)
