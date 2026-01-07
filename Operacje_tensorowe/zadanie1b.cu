#include <cuda_runtime.h>
#include <iostream>

// rozmiar macierzy (kwadratowej)
#define N 16

// rozmiar pojedynczego "tile"
#define TILE_SIZE 4

// kernel GPU działający w obrębie "tile"
__global__ void NaiveMM(const int* A, const int* B, int* C)
{
    // tablice 2D przechowujące wszystkie fragmenty macierzy źródłowych
    // wykorzystywanych w obecnym bloku
    __shared__ int sA[TILE_SIZE][TILE_SIZE];
    __shared__ int sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // każdy wątek będzie przechowywać tutaj wynik kombinacji liniowej wiersza A i kolumny B
    int sum = 0.0f;

    // wyznaczenie liczby "tiles" w całej macierzy wynikowej
    int numTiles = N / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // wczytanie do pamięci tymczasowej wykorzystywanych "tiles" z macierzy źródłowych
        sA[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE_SIZE + threadIdx.x)];
        sB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];

        // synchronizacja wątków oczekująca na zakończenie wczytywania wymaganych "tiles" źródłowych
        __syncthreads();

        // wyznaczenie sum cząstkowych w bieżącym "tile" wynikowym
        for (int k = 0; k < TILE_SIZE; k++)
        {
            C[row * N + col] = sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
    }
}


int main()
{
    int *h_A = new int[N*N];
    int *h_B = new int[N*N];
    int *h_C_gpu = new int[N*N];

    for (int i = 0; i < N*N; i++)
    {
        h_A[i] = static_cast<int>(2);
        h_B[i] = static_cast<int>(2);
    }

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N*N*sizeof(int));
    cudaMalloc(&d_B, N*N*sizeof(int));
    cudaMalloc(&d_C, N*N*sizeof(int));

    cudaMemcpy(d_A, h_A, N*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*N*sizeof(int), cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(N / TILE_SIZE, N / TILE_SIZE);

    // jak wygląda przydział rozmiaru pamięci współdzielonej?
    NaiveMM<<<blocks, threads>>>(d_A, d_B, d_C);

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