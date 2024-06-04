project(cccc-mnist)

set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_FLAGS "-O3 -std=c++17 -pthread -fopenmp")

aux_source_directory(./ SRC_LIST)

include_directories(../include /usr/include /usr/local/include /usr/local/cuda/include ../src ../../mlcc ../cccc-cuda ../cccc)
link_directories(.)

add_library(cccc-mnist SHARED ${SRC_LIST})
target_link_libraries(cccc-mnist cccc)

