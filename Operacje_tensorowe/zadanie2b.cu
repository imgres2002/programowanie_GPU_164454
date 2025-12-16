#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <cmath>
#define N 4

// kernel wyznaczający równolegle każdą z wartości w macierzy wynikowej
__global__ void NaiveMM(float*A, float*C)
{
    __shared__ float sA[N];
    // wiersz oraz kolumna aktualnie pobieranych elementów z macierzy źródłowych
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0;
    float squaredDifferences = 0.0;
    for(int i = 0; i<N;i++)
        sA[threadIdx.x] = A[row * N + (i * N + threadIdx.x)];
    __syncthreads();

    if (row < N && col < N)
    {

        for (int k = 0; k < N; k++) {
            sum += sA[row * N + k];
        }
    }
    float mean = sum/N;
    for (int k = 0; k < N; k++) {
        squaredDifferences += (A[row * N + k] - mean) * (A[row * N + k] - mean);
    }
    float standardDeviation = sqrt(squaredDifferences / N);
    if (row < N && col < N)
    {
        for (int k = 0; k < N; k++) {
            C[row * N + k] = (A[row * N + k] - mean) / standardDeviation;
        }
    }
}

int main()
{
    // inicjalizacja tablic 1D do przechowywania macierzy
    float* h_A = new float[N*N];
    float* h_C = new float[N*N];
    float* h_C_gpu = new float[N*N];

    // tablice operujące po stronie GPU
    float *d_A, *d_C;
    cudaMalloc((void**)&d_A, N*N * sizeof(float));
    cudaMalloc((void**)&d_C, N*N * sizeof(float));

    // wypełnienie tablic wartosciami pseudolosowymi
    for(int i = 0; i<N*N; i++)
    {
//        h_A[i] = std::rand() % 10;
        h_A[i] = i;
        h_C[i] = 0;
        h_C_gpu[i] = 0;
    }

    // transfer macierzy źródłowych do pamięci GPU
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // siatka bloków 2D o rozmiarze 16x16
    dim3 threadsperblock(N,N);

    // 2D siatka wątków w bloku o rozmiarze 32x32
    dim3 numBlocks((N + N - 1) / N, (N + N - 1) / N);

    // wykonanie kerneli po stronie GPU
    NaiveMM<<<numBlocks,threadsperblock>>>(d_A, d_C);

    // transfer wynikowej tablicy do pamięci CPU
    cudaMemcpy(h_C_gpu, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        for (int  j=0; j < N; j++)
        {
            printf("%f ", h_A[i * N + j]);
        }
        std::cout << "|";
        for (int  j=0; j < N; j++)
        {
            printf("%f ", h_C_gpu[i * N + j]);
        }
        std::cout << std::endl;
    }

    // zwolnienie pamięci CPU
    delete[] h_A;
    delete[] h_C;
    delete[] h_C_gpu;

    // zwolnienie pamięci GPU
    cudaFree(d_A);
    cudaFree(d_C);

}