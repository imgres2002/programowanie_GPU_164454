#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void kernel() // kernel, to be executed on GPU
{
	printf("\nThe thread ID is %d", threadIdx.x);
}

int main() // function, execute on CPU
{
	// format: <<<num_of_blocks, num_of_threads_per_block>>>
	kernel << <2, 128 >> > ();

	return 0;
}