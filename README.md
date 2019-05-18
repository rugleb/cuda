[![Build Status](https://travis-ci.com/rugleb/cuda.svg?branch=master)](https://travis-ci.com/rugleb/cuda)
[![Language](https://img.shields.io/badge/Lang-CUDA-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## About

A simple example of a program using the GPU for parallel calculations
on the NVIDIA video card and the CUDA technology.

## Task

Implement a program that implements the heat exchange process in a rod of ARRAY SIZE length
using an explicit difference scheme of the finite difference method.

## Run

1. Compile: `nvcc main.cu`
2. Run: `optirun ./a.out [ARRAY SIZE] [THREADS NUMBER]`

## Performance

CPU: Core i7-6500U CPU @ 2.50GHz × 4  
GPU: GeForce 940M

THREADS NUMBER = 100 (default)

```
gleb@home:~/Projects/cuda$ nvcc main.cu && optirun ./a.out 10
>>> CPU time: 0.003 ms
>>> GPU time: 0.301 ms
>>> Rate : 0.010
gleb@home:~/Projects/cuda$ nvcc main.cu && optirun ./a.out 1000
>>> CPU time: 0.282 ms
>>> GPU time: 0.284 ms
>>> Rate : 0.992
gleb@home:~/Projects/cuda$ nvcc main.cu && optirun ./a.out 10000
>>> CPU time: 3.091 ms
>>> GPU time: 0.427 ms
>>> Rate : 7.233
gleb@home:~/Projects/cuda$ nvcc main.cu && optirun ./a.out 100000
>>> CPU time: 29.232 ms
>>> GPU time: 1.904 ms
>>> Rate : 15.353
```
