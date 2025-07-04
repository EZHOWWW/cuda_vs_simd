cmake_minimum_required(VERSION 3.10)

project(CudaVsSimd VERSION 0.1 LANGUAGES CXX CUDA)

add_subdirectory(example1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "86")


add_compile_options(-O3 -Wall -Wextra)
