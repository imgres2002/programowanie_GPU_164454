#include <cuda_runtime.h>
#include <iostream>

// rozmiar macierzy (kwadratowej)
#define N 8

// rozmiar pojedynczego "tile"
#define TILE_SIZE 4

#define TILES_NUM ((N+N)/TILE_SIZE)
#define TILES_ROWS (N/TILE_SIZE)

__global__ void NaiveMM(const float* A, float* C)
{
    int tileCol = blockIdx.x;
    int tileRow = blockIdx.y;

    // Sprawdzenie, czy nie wychodzimy poza zakres macierzy wynikowej
    if (tileRow < TILES_ROWS && tileCol < TILES_ROWS) {
        float sum = 0.0f;

        // Bezpośrednie sumowanie z pamięci globalnej
        // Pętle przechodzą przez obszar przypisany do danego kafelka
        for (int i = 0; i < TILE_SIZE; i++) {
            for (int j = 0; j < TILE_SIZE; j++) {
                int globalRow = tileRow * TILE_SIZE + i;
                int globalCol = tileCol * TILE_SIZE + j;

                // Odczyt bezpośrednio z tablicy A w VRAM
                sum += A[globalRow * N + globalCol];
            }
        }

        // Zapis wyniku do tablicy C
        int tileIdx = tileRow * TILES_ROWS + tileCol;
        C[tileIdx] = sum;
    }
}

int main()
{
    float *h_A = new float[N*N];
    float *h_C_gpu = new float[TILES_NUM];

    for (int i = 0; i < N*N; i++)
    {
        h_A[i] = i;
    }

    float *d_A, *d_C;
    cudaMalloc(&d_A, N*N*sizeof(float));
    cudaMalloc(&d_C, TILES_NUM*sizeof(float));

    cudaMemcpy(d_A, h_A, N*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(1, 1);
    dim3 blocks(TILES_ROWS, TILES_ROWS);

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
        printf("%f ", h_C_gpu[i]);
    }

    delete[] h_A;

    cudaFree(d_A);
    cudaFree(d_C);

    return 0;
}