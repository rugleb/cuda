#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define TIME                5.
#define TIME_STEP           .1

#define SIZE                (unsigned int) 1e+5
#define STEP                1

#define K                   (double) TIME_STEP / SQUARE(STEP)

#define THREADS             (unsigned int) 1e+2
#define BLOCKS              (unsigned int) SIZE / THREADS

#define SQUARE(x)           (x * x)
#define HANDLE_ERROR(err)   (HandleError(err, __FILE__, __LINE__))


static void HandleError(cudaError_t err, const char *file, int line) 
{
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

__global__ void Kernel(double * device, const unsigned int size)
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

float runCPU(double * host, unsigned int size)
{
    float start = clock();

    for (double t = 0; t < TIME; t += TIME_STEP) {
        host[0] = .0;

        for (int i = 1; i < size - 1; i++) {
            host[i] = (host[i + 1] - 2 * host[i] + host[i - 1]) * K + host[i];
        }

        host[size - 1] = host[size - 2] + 5 * STEP;
    }

    return 1e+3 * (clock() - start) / CLOCKS_PER_SEC;
}

float runGPU(double * device, unsigned int size, unsigned int threads)
{
    float GPUtime;
    cudaEvent_t GPUstart, GPUstop;

	HANDLE_ERROR(cudaEventCreate(&GPUstart));
    HANDLE_ERROR(cudaEventCreate(&GPUstop));

    unsigned int blocks = (unsigned int) size % threads == 0
        ? size / threads
        : size / threads + 1;

    dim3 Threads(threads);
    dim3 Blocks(blocks);
    
    HANDLE_ERROR(cudaEventRecord(GPUstart, 0));

    for (double t = 0; t < TIME; t += TIME_STEP) {
        Kernel <<< Blocks, Threads >>> (device, size);
    }

	HANDLE_ERROR(cudaEventRecord(GPUstop, 0));
	HANDLE_ERROR(cudaEventSynchronize(GPUstop));
    HANDLE_ERROR(cudaEventElapsedTime(&GPUtime, GPUstart, GPUstop));

    HANDLE_ERROR(cudaEventDestroy(GPUstart));
    HANDLE_ERROR(cudaEventDestroy(GPUstop));

    return GPUtime;
}

double * makeHost(unsigned int size)
{
    unsigned int memory = sizeof(double) * size;

    double * host = (double *)malloc(memory);

    for (unsigned int i = 0; i < size; i++) {
        host[i] = .0;
    }

    return host;
}

double * makeDevice(double * host, unsigned int size)
{
    unsigned int memory = sizeof(double) * size;

    double * device;

	HANDLE_ERROR(cudaMalloc((void**)&device, memory));
    HANDLE_ERROR(cudaMemcpy(device, host, memory, cudaMemcpyHostToDevice));

    return device;
}

int main(int argc, char **argv)
{
    unsigned int size = (unsigned int) argc > 1 ? atol(argv[1]) : SIZE;
    unsigned int threads = (unsigned int) argc > 2 ? atol(argv[2]) : THREADS;

    double * host = makeHost(size);
    double * device = makeDevice(host, size);

    float CPUtime = runCPU(host, size);
    float GPUtime = runGPU(device, size, threads);

    printf("CPU time: %.3f ms\n", CPUtime);
    printf("GPU time: %.3f ms\n", GPUtime);
    printf("Rate : %.3f\n", CPUtime / GPUtime);
    
    free(host);
    HANDLE_ERROR(cudaFree(device));
    
    return 0;
}
