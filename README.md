[![Build Status](https://travis-ci.com/rugleb/cuda.svg?branch=master)](https://travis-ci.com/rugleb/cuda)
[![Language](https://img.shields.io/badge/Lang-CUDA-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A simple program that uses parallel GPU computing on an NVIDIA video card using [CUDA](https://developer.nvidia.com/cuda-zone) technology.

The implemented program simulates the process of heat transfer along the entire length of the rod of a given size using an explicit [finite-difference scheme](https://en.wikipedia.org/wiki/Finite_difference_method).  
The longer the rod (array size) - the more calculations need to be done to achieve the result.

## Requirements

* Linux machine
* [CMake 3.1 or later](https://cmake.org/download)
* [CUDA Toolkit 9](https://developer.nvidia.com/cuda-90-download-archive)

## Instructions

1. Compile: `nvcc main.cu`
2. Run: `optirun ./a.out [ARRAY SIZE] [THREADS NUMBER]`

## Performance

CPU: Core i7-6500U CPU @ 2.50GHz ×4  
GPU: GeForce 940M

THREADS NUMBER = 100

```
gleb@home:~/Projects/cuda$ nvcc main.cu && optirun ./a.out 10
>>> CPU time: 0.003 ms
>>> GPU time: 0.301 ms
>>> Rate: 0.010

gleb@home:~/Projects/cuda$ nvcc main.cu && optirun ./a.out 1000
>>> CPU time: 0.282 ms
>>> GPU time: 0.284 ms
>>> Rate: 0.992

gleb@home:~/Projects/cuda$ nvcc main.cu && optirun ./a.out 10000
>>> CPU time: 3.091 ms
>>> GPU time: 0.427 ms
>>> Rate: 7.233

gleb@home:~/Projects/cuda$ nvcc main.cu && optirun ./a.out 100000
>>> CPU time: 29.232 ms
>>> GPU time: 1.904 ms
>>> Rate: 15.353
```

The final graph of the dependence of the performance gain on the size of the array:  

![](https://github.com/rugleb/cuda/blob/master/benchmark/chart.png?raw=true)

## License

This repo is published under the MIT license, see [LICENSE](https://github.com/rugleb/cuda/blob/master/LICENSE).
