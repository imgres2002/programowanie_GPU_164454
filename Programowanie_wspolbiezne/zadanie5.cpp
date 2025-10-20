#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cmath>
#include <filesystem>

std::mutex mtx;


const int VECTOR_LEN = 100;
const int THREAD_COUNT = 12;
const int FRAGMENTS_NUM = THREAD_COUNT / 2;
const int CHUNK_SIZE= VECTOR_LEN/FRAGMENTS_NUM;

int vector[VECTOR_LEN];
int sums[FRAGMENTS_NUM];
int averages[FRAGMENTS_NUM];

void sumVectorElements(int chunkNum) {
    int start = chunkNum * CHUNK_SIZE;
    int end = start + CHUNK_SIZE;
    int sum = 0;
    for (int i = start; i < end; ++i)
        sum += vector[i];
    sums[chunkNum] = sum;
}

void averageVectorElements(int chunkNum) {
    int start = chunkNum * CHUNK_SIZE;
    int end = start + CHUNK_SIZE;
    int sum = 0;
    for (int i = start; i < end; ++i)
        sum += vector[i];
    float average = sum / CHUNK_SIZE;
    averages[chunkNum] = average;
}

void createVector(int* vec) {
    for (int i = 0; i < VECTOR_LEN; ++i) {
        vec[i] = rand() % 10 + 1;
    }
}

int main() {
    createVector(vector);

    std::thread sumThreads[FRAGMENTS_NUM];
    std::thread averageThreads[FRAGMENTS_NUM];

    for (int i = 0; i < FRAGMENTS_NUM; ++i) {
        sumThreads[i] = std::thread(sumVectorElements, i);
        averageThreads[i] = std::thread(averageVectorElements, i);
    }

    for (int i = 0; i < FRAGMENTS_NUM; ++i) {
        sumThreads[i].join();
        averageThreads[i].join();
    }
    int sum = 0;
    float average = 0;
    for (int i = 0; i < FRAGMENTS_NUM; ++i) {
        sum += sums[i];
        average += averages[i];
    }
    average = average / FRAGMENTS_NUM;

    std::cout << "suma: " << sum << "\n"
              << "średnia: " << average;

    return 0;
}
