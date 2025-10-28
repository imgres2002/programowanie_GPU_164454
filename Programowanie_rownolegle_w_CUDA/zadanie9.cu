#include "cuda_runtime.h"
#include <iostream>

#define SIZE 4100
#define CHUNK_SIZE 1024

// kernel sumujacy wektory element po elemencie
__global__ void vectorAdd(int* A, int* B, int* C, int n)
{
    int i = threadIdx.x;
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
    int chunkSize = CHUNK_SIZE * sizeof(int);

    // krok 2. alokacja wektorów w pamięci RAM CPU
    A = (int*)malloc(size);
    B = (int*)malloc(size);
    C = (int*)malloc(size);

    // krok 3. alokacja pamięci RAM GPU
    cudaMalloc((void**)&dC, chunkSize);
    cudaMalloc((void**)&dA, chunkSize);
    cudaMalloc((void**)&dB, chunkSize);

    // krok 4. inicjalizacja wartości wektorów
    for (int i = 0; i < SIZE; i++)
    {
        A[i] = i;
        B[i] = SIZE - i;
    }

    for (int i = 0; i < SIZE-CHUNK_SIZE; i += CHUNK_SIZE) {
        // krok 5. transfer wektorów z pamięci RAM CPU do pamięci GPU
        cudaMemcpy(dA, &A[i], chunkSize, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, &B[i], chunkSize, cudaMemcpyHostToDevice);

        // krok 6. uruchomienie kernela
        int blocks = (CHUNK_SIZE + 1024 - 1) / 1024;
        vectorAdd <<<blocks, 1024>>> (dA, dB, dC, CHUNK_SIZE);
        cudaDeviceSynchronize();

        // krok 7. transfer wektora wynikowego z pamięci GPU do CPU
        cudaMemcpy(&C[i], dC, chunkSize, cudaMemcpyDeviceToHost);
    }
    int reminder = SIZE % CHUNK_SIZE;
    int last_chunk = SIZE - reminder;
    cudaMemcpy(dA, &A[last_chunk], chunkSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, &B[last_chunk], chunkSize, cudaMemcpyHostToDevice);
    int blocks = (CHUNK_SIZE + 1024 - 1) / 1024;
    vectorAdd <<<blocks, 1024>>> (dA, dB, dC, reminder);
    cudaDeviceSynchronize();
    cudaMemcpy(&C[last_chunk], dC, reminder * sizeof(int), cudaMemcpyDeviceToHost);

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
