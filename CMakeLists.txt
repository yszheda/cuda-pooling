cmake_minimum_required(VERSION 3.24)
project(cuda_pooling LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80;90;100;110)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
)
FetchContent_MakeAvailable(pybind11)

add_library(pooling_kernels STATIC
    src/pooling_max.cu
    src/pooling_avg.cu
)
target_include_directories(pooling_kernels PUBLIC include)
target_compile_options(pooling_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

find_package(CUDAToolkit REQUIRED)

pybind11_add_module(_pooling src/pybind_module.cpp)
target_link_libraries(_pooling PRIVATE pooling_kernels Python3::NumPy CUDA::cudart)
target_compile_options(_pooling PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
