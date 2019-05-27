#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define TIME                5.
#define TIME_STEP           .1

#define STEP                1.
#define K                   TIME_STEP / SQUARE(STEP)

#define SQUARE(x)           (x * x)
#define HANDLE_ERROR(err)   (HandleError(err, __FILE__, __LINE__))


static void HandleError(cudaError_t err, const char *file, uint line)
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

__global__ void Kernel(double * device, const uint size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i == 0) {
        device[i] = .0;
    } else if (i == size - 1) {
        device[size - 1] = device[size - 2] + 5 * 1;
    } else if (i < size) {
        device[i] = (device[i + 1] - 2 * device[i] + device[i - 1]) * K + device[i];
    }
}

float runCPU(double * host, uint size)
{
    float start = clock();

    for (double t = 0.; t < TIME; t += TIME_STEP) {
        host[0] = .0;

        for (int i = 1; i < size - 1; i++) {
            host[i] = (host[i + 1] - 2 * host[i] + host[i - 1]) * K + host[i];
        }

        host[size - 1] = host[size - 2] + 5 * STEP;
    }

    return 1e+3 * (clock() - start) / CLOCKS_PER_SEC;
}

float runGPU(double * device, uint size, uint threads)
{
    float time;
    cudaEvent_t start, stop;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    uint blocks = (uint) size % threads == 0
        ? size / threads
        : size / threads + 1;

    dim3 Threads(threads);
    dim3 Blocks(blocks);
    
    HANDLE_ERROR(cudaEventRecord(start, 0));

    for (double t = 0; t < TIME; t += TIME_STEP) {
        Kernel <<< Blocks, Threads >>> (device, size);
    }

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));

    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    return time;
}

double * makeHost(uint size)
{
    uint memory = sizeof(double) * size;

    double * host = (double *)malloc(memory);

    for (uint i = 0; i < size; i++) {
        host[i] = .0;
    }

    return host;
}

double * makeDevice(double * host, uint size)
{
    uint memory = sizeof(double) * size;

    double * device;

    HANDLE_ERROR(cudaMalloc((void**)&device, memory));
    HANDLE_ERROR(cudaMemcpy(device, host, memory, cudaMemcpyHostToDevice));

    return device;
}

int main(int argc, char **argv)
{
    uint size = (uint) argc > 1 ? atol(argv[1]) : 1e+5;
    uint threads = (uint) argc > 2 ? atol(argv[2]) : 1e+2;

    double * host = makeHost(size);
    double * device = makeDevice(host, size);

    float timeCPU = runCPU(host, size);
    float timeGPU = runGPU(device, size, threads);

    printf("CPU time: %.3f ms\n", timeCPU);
    printf("GPU time: %.3f ms\n", timeGPU);
    printf("Rate : %.3f\n", timeCPU / timeGPU);
    
    free(host);
    HANDLE_ERROR(cudaFree(device));
    
    return 0;
}
