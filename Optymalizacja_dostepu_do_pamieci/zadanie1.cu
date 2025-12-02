#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdio.h>
#include <cstdio>

#define a 2

__global__ void MultiplyVectorByScalar(float* input, int n) {
    __shared__ float shared[256];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    shared[tid] = input[index];

    __syncthreads();

    int start_idx = blockIdx.x * blockDim.x;
    for(int i = 0; i < 256; i++){
        input[start_idx + i] = shared[i] * a;
    }
}

int main() {
    int n = 1024;
    size_t bytes = n * sizeof(float);

    float* h_input = new float[n];
    float* d_output = new float[n];
    float* d_input;

    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>(i + 1);
    }

    cudaMalloc(&d_input, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = n / block_size;
    // liczba elementow zgodna z liczba warp'ow w bloku
    size_t shared_mem_size = (block_size / 32) * sizeof(float);

    MultiplyVectorByScalar <<<grid_size, block_size, shared_mem_size >>> (d_input, n);
    cudaDeviceSynchronize();
    cudaMemcpy(d_output, d_input, n * sizeof(float), cudaMemcpyDeviceToHost);


    for (int i = 0; i < n; i++)
    {
        printf("%f * %d = %f\n", h_input[i], a, d_output[i]);
    }

    cudaFree(d_input);
    delete[] h_input;
    delete[] d_output;

    return 0;
}
