#include "cuda_runtime.h"
#include <stdio.h>

#define SIZE 1024

// kernel sumujacy wektory element po elemencie
__global__ void vectorAdd(int* A, int* B, int* C, int n)
{
    int i = threadIdx.x;
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
    vectorAdd <<<1, SIZE>>> (dA, dB, dC, size);

    // krok 7. transfer wektora wynikowego z pamięci GPU do CPU
    cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);

    // krok 8. zwrócenie wyników do wyjścia standardowego
    for (int i = 0; i < SIZE; i++)
    {
        printf("%d + %d = %d\n", A[i], B[i], C[i]);
    }

    // krok 9. wyczyszczenie pamięci GPU oraz CPU
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(A);
    free(B);
    free(C);

    return 0;
}
