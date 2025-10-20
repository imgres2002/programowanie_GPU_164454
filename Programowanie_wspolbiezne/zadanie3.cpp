#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cmath>

std::mutex mtx;


const int VECTOR_LEN = 1000000;
const int THREAD_COUNT = 12;
const int CHUNK_SIZE= VECTOR_LEN/THREAD_COUNT;

int vector1[VECTOR_LEN];
int vector2[VECTOR_LEN];
int vectorsSum[VECTOR_LEN];

void sumVectors(int chunkNum) {
    int start = chunkNum * CHUNK_SIZE;
    int end = start + CHUNK_SIZE;
    for (int i = start; i < end; ++i)
        vectorsSum[i] = vector1[i] + vector2[i];
}

void createVector(int* vec) {
    for (int i = 0; i < VECTOR_LEN; ++i) {
        vec[i] = rand() % 10 + 1;
    }
}

int main() {
    createVector(vector1);
    createVector(vector2);

    std::thread threads[THREAD_COUNT];

    for (int i = 0; i < THREAD_COUNT; ++i) {
        threads[i] = std::thread(sumVectors, i);
    }

    for (int i = 0; i < THREAD_COUNT; ++i) {
        threads[i].join();
    }

    std::cout << "vector1[0] + vector2[0] = "
              << vector1[0] << " + " << vector2[0]
              << " = " << vectorsSum[0] << "\n";

    return 0;
}
