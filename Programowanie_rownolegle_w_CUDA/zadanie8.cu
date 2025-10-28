#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define SIZE 64

__global__ void vectorAddSub(int* A, int* B, int* C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int warpIdValue = 0;
    warpIdValue = threadIdx.x / 32;

    if (warpIdValue % 2 == 0) {
        C[i] = A[i] + B[i];
    } else {
        C[i] = A[i] - B[i];
    }
}

int main()
{
    int* A, * B, * C;
    int* dA, * dB, * dC;
    int size = SIZE * sizeof(int);

    A = (int*)malloc(size);
    B = (int*)malloc(size);
    C = (int*)malloc(size);

    cudaMalloc((void**)&dA, size);
    cudaMalloc((void**)&dB, size);
    cudaMalloc((void**)&dC, size);

    for (int i = 0; i < SIZE; i++)
    {
        A[i] = i;
        B[i] = i;
    }
    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);

    vectorAddSub <<<2, 64>>> (dA, dB, dC, SIZE);

    cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < SIZE/2+1; i+=2)
    {
        printf("%d + %d = %d\n", A[i], B[i], C[i]);
    }

    for (int i = 1; i < SIZE/2; i+=2)
    {
        printf("%d - %d = %d\n", A[i], B[i], C[i]);
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(A);
    free(B);
    free(C);

    return 0;
}

