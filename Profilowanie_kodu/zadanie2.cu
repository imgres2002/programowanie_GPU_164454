#include "cuda_runtime.h"
#include <stdio.h>
#include <iostream>
#include <cmath>

#define SIZE 32768

// kernel sumujacy wektory element po elemencie
__global__ void vectorAdd(int* A, int* B, int* C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    C[i] = A[i] + B[i];
}

int main()
{
    // krok 1. alokacja przestrzeni w pamięci RAM CPU
    int* A, * B, * C;
    int* dA, * dB, * dC;
    int size = SIZE * sizeof(int);

    // krok 2. alokacja wektorów w pamięci RAM CPU
    A = (int*)malloc(size);
    B = (int*)malloc(size);
    C = (int*)malloc(size);

    // krok 3. alokacja pamięci RAM GPU
    cudaMalloc((void**)&dA, size);
    cudaMalloc((void**)&dB, size);
    cudaMalloc((void**)&dC, size);

    // krok 4. inicjalizacja wartości wektorów
    for (int i = 0; i < SIZE; i++)
    {
        A[i] = i;
        B[i] = SIZE - i;
    }

    // krok 5. transfer wektorów z pamięci RAM CPU do pamięci GPU
    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);

    // krok 6. uruchomienie kernela
    int threads_num = 96;
    int blocks_num = static_cast<int>(std::ceil(static_cast<double>(SIZE) / threads_num));
    vectorAdd <<<blocks_num, threads_num>>> (dA, dB, dC, size);

    // krok 7. transfer wektora wynikowego z pamięci GPU do CPU
    cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);

    // krok 9. wyczyszczenie pamięci GPU oraz CPU
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(A);
    free(B);
    free(C);

    return 0;
}
