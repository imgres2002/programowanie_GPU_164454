#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define SIZE 2048

__global__ void vectorSubtract(int* A, int* B, int* C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    C[i] = A[i] - B[i];
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

    vectorSubtract <<<64, 32>>> (dA, dB, dC, SIZE);

    cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < SIZE; i++) {
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
