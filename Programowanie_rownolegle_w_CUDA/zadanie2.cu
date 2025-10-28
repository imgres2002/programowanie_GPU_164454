#include <iostream>
#include "cuda_runtime.h"


#define SIZE 1024

__global__ void vectorMultiply(int* A, int factor, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int warpIdValue = 0;
	warpIdValue = threadIdx.x / 32;
	if (i >= n) {
		return;
	}
	A[i] = warpIdValue * factor;
}

int main()
{
	// krok 1. alokacja przestrzeni w pami�ci RAM CPU
	int* A;
	int* dA;
	int factor = 2;
	int size = SIZE * sizeof(int);
	// krok 2. alokacja wektor�w w pami�ci RAM CPU
	A = (int*)malloc(size);

	// krok 3. alokacja pami�ci RAM GPU
	cudaMalloc((void**)&dA, size);

	// krok 6. uruchomienie kernela
	vectorMultiply << <1, 1024 >> > (dA, factor, SIZE);

	// krok 7. transfer wektora wynikowego z pami�ci GPU do CPU
	cudaMemcpy(A, dA, size, cudaMemcpyDeviceToHost);

	// krok 8. zwr�cenie wynik�w do wyj�cia standardowego
	for (int i = 0; i < SIZE; i++) {
		printf("%d", A[i]);
	}

	// krok 9. wyczyszczenie pami�ci GPU oraz CPU
	cudaFree(dA);
	free(A);

	return 0;
}
