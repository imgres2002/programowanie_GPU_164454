#include "cuda_runtime.h"
#include <stdio.h>

__global__ void vectorCheck()
{
    int warpIdValue = 0;
    warpIdValue = threadIdx.x / 32;
    int sum = threadIdx.x + blockIdx.x + warpIdValue;

    if (sum % 2 == 0) {
        printf("(%d + %d + %d) %% 2 == 0\n", threadIdx.x, blockIdx.x, warpIdValue);
    } else {
        printf("(%d + %d + %d) %% 2 != 0\n", threadIdx.x, blockIdx.x, warpIdValue);
    }
}

int main()
{
    printf("threadIdx.x + blockIdx.x + warpIdValue\n");

    vectorCheck << <4, 32 >> > ();

    return 0;
}
