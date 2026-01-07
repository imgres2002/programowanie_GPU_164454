#include <cuda_runtime.h>
#include <iostream>

// rozmiar macierzy (kwadratowej)
#define N 8

// kernel GPU działający w obrębie "tile"
__global__ void NaiveMM(const int* A, const int* B, int* C)
{
    int row = threadIdx.y;
    int col = threadIdx.x;
    C[blockDim.x * row + col] = A[blockDim.x * row + col] * B[blockDim.x * row + col];
}


int main()
{
    int *h_A = new int[N*N];
    int *h_B = new int[N*N];
    int *h_C_gpu = new int[N*N];

    for (int i = 0; i < N*N; i++)
    {
        h_A[i] = static_cast<int>(i);
        h_B[i] = static_cast<int>(N-1);
    }

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N*N*sizeof(int));
    cudaMalloc(&d_B, N*N*sizeof(int));
    cudaMalloc(&d_C, N*N*sizeof(int));

    cudaMemcpy(d_A, h_A, N*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*N*sizeof(int), cudaMemcpyHostToDevice);

    dim3 threads(N, N);

    // jak wygląda przydział rozmiaru pamięci współdzielonej?
    NaiveMM<<<1, threads>>>(d_A, d_B, d_C);

    cudaMemcpy(h_C_gpu, d_C, N*N*sizeof(int), cudaMemcpyDeviceToHost);


    for (int i = 0; i < N; i++)
    {
        for (int  j=0; j < N; j++)
        {
            printf("%d ", h_A[i * N + j]);
        }
        std::cout << "|";
        for (int  j=0; j < N; j++)
        {
            printf("%d ", h_B[i * N + j]);
        }
        std::cout << "|";
        for (int  j=0; j < N; j++)
        {
            printf("%d ", h_C_gpu[i * N + j]);
        }
        std::cout << std::endl;
    }

    delete[] h_A;
    delete[] h_B;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}