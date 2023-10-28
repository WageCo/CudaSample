cmake_minimum_required(VERSION 3.10)
project(CudaSample)

enable_language(CUDA)

# local Gpu is RTX1050Ti
set(CMAKE_CUDA_ARCHITECTURES "61")

add_executable(${PROJECT_NAME} main.cu)
