project(libwill)
cmake_minimum_required(VERSION 2.6)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

option(CCCC_CPU "CPU support" ON)
 
if(NOT CCCC_CPU)
add_subdirectory(will-cuda)
endif()
add_subdirectory(will)
add_subdirectory(will-windows)
add_subdirectory(will-mnist)

if(WITH_PYTHON)
add_subdirectory(will-python)
endif()
