project(will-windows)
cmake_minimum_required(VERSION 2.6)

aux_source_directory(./ SRC_LIST)

set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_COMPILER $ENV{CXX})
# set(CMAKE_CXX_FLAGS "-O3 -std=c++17 -pthread -fopenmp -DNDEBUG")

if(CCCC_CPU)
set(CMAKE_CXX_FLAGS "-O3 -std=c++17 -pthread -fopenmp -DNO_CUDA -DNDEBUG")
set(LINK_LIBRARIES  openblas)
else()
set(CMAKE_CXX_FLAGS "-O3 -std=c++17 -pthread -fopenmp -DNDEBUG")
set(LINK_LIBRARIES will-cuda cudnn cudart cublas openblas)
endif()
include_directories(/usr/local/cuda/include /usr/local/cuda/include/crt ../common ../will-cuda ../include ../will)
link_directories(/usr/local/cuda/lib64)

add_executable(will-windows ${SRC_LIST})
target_link_libraries(will-windows will)

install(TARGETS will-windows RUNTIME DESTINATION bin)
