#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define SIZE 2048

__global__ void vectorAdd(int* A, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    A[i] += threadIdx.x + blockIdx.x;
}

int main()
{
    int* A;
    int* dA;
    int size = SIZE * sizeof(int);

    A = (int*)malloc(size);

    cudaMalloc((void**)&dA, size);

    for (int i = 0; i < SIZE; i++) {
        A[i] = 100;
    }
    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);

    vectorAdd <<<64, 32>>> (dA, SIZE);

    cudaMemcpy(A, dA, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < SIZE; i++) {
        printf("%d\n", A[i]);
    }

    cudaFree(dA);
    free(A);

    return 0;
}
