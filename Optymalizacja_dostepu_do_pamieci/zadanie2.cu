#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdio.h>
#include <cstdio>

__global__ void AddVectors(float* input_vector1, float*input_vector2, int n) {
    __shared__ float shared[256];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    shared[tid] = input_vector1[index];
    shared[128 + tid] = input_vector2[index];

    __syncthreads();

    int start_idx = blockIdx.x * blockDim.x;
    for(int i = 0; i < 128; i++){
        input_vector1[start_idx + i] = shared[i] + shared[128 + i];
    }
}

int main() {
    int n = 1024;
    size_t bytes = n * sizeof(float);

    float* h_input_vector1 = new float[n];
    float* h_input_vector2 = new float[n];
    float* d_output_vector = new float[n];
    float* d_input_vector1;
    float* d_input_vector2;

    for (int i = 0; i < n; i++) {
        h_input_vector1[i] = static_cast<float>(i + 1);
        h_input_vector2[i] = static_cast<float>(n - 1 - i);
    }

    cudaMalloc(&d_input_vector1, bytes);
    cudaMalloc(&d_input_vector2, bytes);

    cudaMemcpy(d_input_vector1, h_input_vector1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_vector2, h_input_vector2, bytes, cudaMemcpyHostToDevice);

    int block_size = 128;
    int grid_size = n / block_size;
    // liczba elementow zgodna z liczba warp'ow w bloku
    size_t shared_mem_size = (block_size * 2 / 32) * sizeof(float);

    AddVectors <<<grid_size, block_size, shared_mem_size >>> (d_input_vector1, d_input_vector2, n);
    cudaDeviceSynchronize();
    cudaMemcpy(d_output_vector, d_input_vector1, n * sizeof(float), cudaMemcpyDeviceToHost);


    for (int i = 0; i < n; i++)
    {
        printf("%f + %f = %f\n", h_input_vector1[i], h_input_vector2[i], d_output_vector[i]);
    }

    cudaFree(d_input_vector1);
    cudaFree(d_input_vector2);
    delete[] h_input_vector1;
    delete[] h_input_vector2;
    delete[] d_output_vector;

    return 0;
}
