#include "cuda_runtime.h"
#include <stdio.h>

#define SIZE 1024

// kernel sumujacy wektory element po elemencie
__global__ void vectorCheck(int* A)
{
	int i = threadIdx.x;
	if (i % 3 == 0) {
		A[i] = 1;
	}
	else
	{
		A[i] = 0;
	}

}

int main()
{
	// krok 1. alokacja przestrzeni w pamiêci RAM CPU
	int* A;
	int* dA;
	int size = SIZE * sizeof(int);

	// krok 2. alokacja wektorów w pamiêci RAM CPU
	A = (int*)malloc(size);

	// krok 3. alokacja pamiêci RAM GPU
	cudaMalloc((void**)&dA, size);

	// krok 4. inicjalizacja wartoœci wektorów
	for (int i = 0; i < SIZE; i++)
	{
		A[i] = i;
	}

	// krok 5. transfer wektorów z pamiêci RAM CPU do pamiêci GPU
	cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);

	// krok 6. uruchomienie kernela
	vectorAdd << <1, 1024 >> > (dA, size);

	// krok 7. transfer wektora wynikowego z pamiêci GPU do CPU
	cudaMemcpy(A, dA, size, cudaMemcpyDeviceToHost);

	// krok 8. zwrócenie wyników do wyjœcia standardowego
	for (int i = 0; i < SIZE; i++)
	{
		if (A[i] == 0) {
			printf("W¹tek %d jest podzielny przez 3\n", i);
		}
	}

	// krok 9. wyczyszczenie pamiêci GPU oraz CPU
	cudaFree(dA);
	free(A);

	return 0;
}
