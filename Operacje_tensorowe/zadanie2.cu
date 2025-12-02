#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>

// kernel wyznaczający równolegle każdą z wartości w macierzy wynikowej
__global__ void NaiveMM(float*A, float*B, float*C, int N)
{
    // wiersz oraz kolumna aktualnie pobieranych elementów z macierzy źródłowych
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // jeżeli indeksy nie wychodzą poza zakres macierzy
    if (row < N && col < N)
    {
        for (int k = 0; k < N; k++) {
            C[row * N + col] = A[row * N + col] * B[row * N + col];
        }
    }
}

// implementacja sekwencyjna - CPU
void mmCpu(const float* A, const float* B, float* C, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0;
            for (int k = 0; k < N; k++)
            {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}


int main()
{
    // wymiar macierzy (kwadratowych)
    const int N = 512;

    // inicjalizacja tablic 1D do przechowywania macierzy
    float* h_A = new float[N*N];
    float* h_B = new float[N*N];
    float* h_C = new float[N*N];
    float* h_C_gpu = new float[N*N];

    // tablice operujące po stronie GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N*N * sizeof(float));
    cudaMalloc((void**)&d_B, N*N * sizeof(float));
    cudaMalloc((void**)&d_C, N*N * sizeof(float));

    // wypełnienie tablic wartosciami pseudolosowymi
    for(int i = 0; i<N*N; i++)
    {
//        h_A[i] = std::rand() % 10;
        h_A[i] = 2;
//        h_B[i] = std::rand() % 10;
        h_B[i] = 2;
        h_C[i] = 0;
        h_C_gpu[i] = 0;
    }

    // wykonanie iloczynu po stronie CPU
    mmCpu(h_A, h_B, h_C, N);

    // transfer macierzy źródłowych do pamięci GPU
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 16;

    // siatka bloków 2D o rozmiarze 16x16
    dim3 threadsperblock(block_size,block_size);

    // 2D siatka wątków w bloku o rozmiarze 32x32
    dim3 numBlocks((N + block_size - 1) / block_size, (N + block_size - 1) / block_size);

    // wykonanie kerneli po stronie GPU
    NaiveMM<<<numBlocks,threadsperblock>>>(d_A, d_B, d_C, N);

    // transfer wynikowej tablicy do pamięci CPU
    cudaMemcpy(h_C_gpu, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << h_C_gpu[i * N + j] << " ";
        }

        // Newline for new row
        std::cout << std::endl;
    }

    std::cout << h_C << std::endl;
    std::cout << h_C_gpu << std::endl;

    // zwolnienie pamięci CPU
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_gpu;

    // zwolnienie pamięci GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}