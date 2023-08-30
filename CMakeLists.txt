project(libcccc)
cmake_minimum_required(VERSION 2.6)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

option(CCCC_CPU "CPU support" ON)
 
if(NOT CCCC_CPU)
    add_subdirectory(cccc-cuda)
else()
    add_definitions(-D NO_CUDA=1)
endif()
add_subdirectory(cccc)
add_subdirectory(cccc-windows)
add_subdirectory(cccc-mnist)

if(WITH_PYTHON)
    add_subdirectory(cccc-python)
endif()
