project(woco-python)
cmake_minimum_required(VERSION 2.6)

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})
find_package(PythonLibs)

set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_FLAGS "-O3 -std=c++17 -pthread -fopenmp")

include_directories(../include /usr/local/cuda/include ../woco ../../common ../woco-cuda ${PYTHON_INCLUDE_PATH})

set(CMAKE_SWIG_FLAGS "")
set_property(SOURCE woco.i PROPERTY CPLUSPLUS ON)
#swig_add_library(woco TYPE SHARED LANGUAGE python SOURCES woco.i)
swig_add_module(woco python woco.i)
swig_link_libraries(woco woco ${PYTHON_LIBRARIES})
