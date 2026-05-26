cmake_minimum_required(VERSION 3.18)
project(taskflow_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12)

set(TF_BUILD_CUDA ON CACHE BOOL "" FORCE)
set(TF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TF_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(/home/kobayashi/taskflow taskflow-build)

add_executable(run
  taskflow_stg_cuda_demo.cu
  leveling.cpp
)

target_include_directories(run PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  /usr/local/cuda/include
)

target_compile_definitions(run PRIVATE TF_ENABLE_CUDA)

target_link_libraries(run PRIVATE
  Taskflow
  CUDA::cudart
)

set_target_properties(run PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES 90
)