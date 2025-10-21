#include "cuda_runtime.h"
#include <stdio.h>

#define SIZE 1024

// kernel sumujacy wektory element po elemencie
__global__ void vectorAdd(int* A, int* C, int factor, int n)
{
	int i = threadIdx.x;
	C[i] = A[i] * factor;
}

int main()
{
	// krok 1. alokacja przestrzeni w pami�ci RAM CPU
	int* A, * C;
	int* dA, * dC;
	int factor;
	factor = 2;
	int size = SIZE * sizeof(int);

	// krok 2. alokacja wektor�w w pami�ci RAM CPU
	A = (int*)malloc(size);
	C = (int*)malloc(size);

	// krok 3. alokacja pami�ci RAM GPU
	cudaMalloc((void**)&dA, size);
	cudaMalloc((void**)&dC, size);

	// krok 4. inicjalizacja warto�ci wektor�w
	for (int i = 0; i < SIZE; i++)
	{
		A[i] = i;
	}

	// krok 5. transfer wektor�w z pami�ci RAM CPU do pami�ci GPU
	cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);

	// krok 6. uruchomienie kernela
	vectorAdd << <1, 1024 >> > (dA, dC, factor, size);

	// krok 7. transfer wektora wynikowego z pami�ci GPU do CPU
	cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);

	// krok 8. zwr�cenie wynik�w do wyj�cia standardowego
	for (int i = 0; i < SIZE; i++)
	{
		printf("%d * %d = %d\n", A[i], factor, C[i]);
	}

	// krok 9. wyczyszczenie pami�ci GPU oraz CPU
	cudaFree(dA);
	cudaFree(dC);
	free(A);
	free(C);

	return 0;
}
