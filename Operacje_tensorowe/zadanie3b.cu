#include <cuda_runtime.h>
#include <iostream>

// rozmiar macierzy (kwadratowej)
#define N 8

// rozmiar pojedynczego "tile"
#define TILE_SIZE 4
//int tiles_num = N+N/TILE_SIZE;
#define TILES_NUM ((N+N)/TILE_SIZE)
#define TILES_ROWS (N/TILE_SIZE)

// kernel GPU działający w obrębie "tile"
__global__ void NaiveMM(const float* A, float* C)
{
    // tablice 2D przechowujące wszystkie fragmenty macierzy źródłowych
    // wykorzystywanych w obecnym bloku
    __shared__ float sA[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // każdy wątek będzie przechowywać tutaj wynik kombinacji liniowej wiersza A i kolumny B
    float sum = 0.0f;

    // wyznaczenie liczby "tiles" w całej macierzy wynikowej
    int numTiles = N / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // wczytanie do pamięci tymczasowej wykorzystywanych "tiles" z macierzy źródłowych
        sA[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE_SIZE + threadIdx.x)];

        // synchronizacja wątków oczekująca na zakończenie wczytywania wymaganych "tiles" źródłowych
        __syncthreads();

        // wyznaczenie sum cząstkowych w bieżącym "tile" wynikowym
        for (int k = 0; k < TILE_SIZE; k++)
        {
            sum += sA[threadIdx.y][k];
        }

        // oczekiwanie na zakończenie wyznaczania sum
        __syncthreads();
    }

    int C_row = (blockIdx.y * TILE_SIZE + threadIdx.y) / TILES_NUM;
    int C_col = (blockIdx.x * TILE_SIZE + threadIdx.x) / TILES_NUM;
    printf("row %d \n", C_row);
    printf("col: %d \n", C_col);
    printf("sum: %d \n", sum);
    // zapisanie wyniku w wektorze (imitującym macierz) C
    C[C_row * TILES_ROWS + C_col] = sum;
}

int main()
{
    float *h_A = new float[N*N];
    float *h_C = new float[TILES_NUM];
    float *h_C_gpu = new float[TILES_NUM];

    for (int i = 0; i < N*N; i++)
    {
        h_A[i] = i;
    }

    float *d_A, *d_C;
    cudaMalloc(&d_A, N*N*sizeof(float));
    cudaMalloc(&d_C, TILES_NUM*sizeof(float));

    cudaMemcpy(d_A, h_A, N*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(N / TILE_SIZE, N / TILE_SIZE);

    // jak wygląda przydział rozmiaru pamięci współdzielonej?
    NaiveMM<<<blocks, threads>>>(d_A, d_C);

    cudaMemcpy(h_C_gpu, d_C, TILES_NUM*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        for (int  j=0; j < N; j++)
        {
            printf("%f ", h_A[i * N + j]);
        }
        std::cout << std::endl;
    }

    for (int i = 0; i < TILES_NUM; i++)
    {
        for (int  j=0; j < TILES_NUM; j++)
        {
            printf("%f ", h_C_gpu[i * TILES_NUM + j]);
        }
        std::cout << std::endl;
    }

    delete[] h_A;
    delete[] h_C;

    cudaFree(d_A);
    cudaFree(d_C);

    return 0;
}