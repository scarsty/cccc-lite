project(woco-sln)
cmake_minimum_required(VERSION 2.6)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(woco-cuda)
add_subdirectory(woco)
add_subdirectory(woco-windows)
add_subdirectory(woco-mnist)

if(WITH_PYTHON)
add_subdirectory(woco-python)
endif()
