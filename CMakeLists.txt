project(libcccc)
cmake_minimum_required(VERSION 2.6)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(CCCC_CUDA)
    add_subdirectory(cccc-cuda)
    add_definitions(-DENABLE_CUDA=1)
else()
    add_definitions(-DENABLE_CUDA=0)
endif()
if(CCCC_HIP)
    add_subdirectory(cccc-hip)
    add_definitions(-DENABLE_HIP=1)
else()
    add_definitions(-DENABLE_HIP=0)
endif()

add_subdirectory(cccc)
add_subdirectory(cccc-windows)
add_subdirectory(cccc-mnist)

