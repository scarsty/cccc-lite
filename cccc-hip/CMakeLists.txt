project(cccc-hip)
cmake_minimum_required(VERSION 2.6)


set(GPU_RUNTIME "HIP" CACHE STRING "Switches between HIP and CUDA")

enable_language(${GPU_RUNTIME})
set(CMAKE_${GPU_RUNTIME}_STANDARD 17)
set(CMAKE_${GPU_RUNTIME}_EXTENSIONS OFF)
set(CMAKE_${GPU_RUNTIME}_STANDARD_REQUIRED ON)

if(WIN32)
    set(ROCM_ROOT "$ENV{HIP_PATH}" CACHE PATH "Root directory of the ROCm installation")
else()
    set(ROCM_ROOT "/opt/rocm" CACHE PATH "Root directory of the ROCm installation")
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")

add_library(cccc-hip SHARED hip_functions.hip)
# Make example runnable using ctest
set(include_dirs "../../Common")
if(GPU_RUNTIME STREQUAL "CUDA")
    list(APPEND include_dirs "${ROCM_ROOT}/include")
endif()

#target_include_directories(${project_name} PRIVATE ${include_dirs})
set_source_files_properties(hip_functions.hip PROPERTIES LANGUAGE ${GPU_RUNTIME})

install(TARGETS cccc-hip)


