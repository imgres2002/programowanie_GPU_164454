#include "cuda_runtime.h"
#include <stdio.h>

__global__ void vectorCheck()
{
	int i = threadIdx.x;
	if (i % 3 == 0) {
		printf("Wątek %d jest podzielny przez 3\n", i);
	} else {
		printf("Wątek %d NIE jest podzielny przez 3\n", i);
	}
}

int main()
{
	vectorCheck << <2, 32 >> > ();

	return 0;
}
