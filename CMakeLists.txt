cmake_minimum_required(VERSION 3.1)
find_package(CUDA 9 REQUIRED)

cuda_add_executable(main main.cu)
