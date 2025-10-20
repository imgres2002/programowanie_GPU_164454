#include <iostream>
#include <thread>
#include <mutex>

const int VECTOR_LEN = 10;

int vector1[VECTOR_LEN];
int vector2[VECTOR_LEN];
int vectorSum = 0;
int vectorProduct = 1;

int sumVectorElements() {
    for (int i = 0; i < VECTOR_LEN; ++i) {
        vectorSum += vector1[i];
    }
    return vectorSum;
}

int multiplyVectorElements() {
    for (int i = 0; i < VECTOR_LEN; ++i) {
        vectorProduct = vectorProduct * vector2[i];
    }
    return vectorProduct;
}

void createVector(int* vec) {
    for (int i = 0; i < VECTOR_LEN; ++i) {
        vec[i] = (rand() % 5) + 1;
        std::cout << vec[i] << " ";
    }
}

int main() {
    createVector(vector1);
    createVector(vector2);

    // tworzenie dwóch wątków
    std::thread thread1(sumVectorElements);
    std::thread thread2(multiplyVectorElements);

    // oczekiwanie na zakończenie wątków
    thread1.join();
    thread2.join();

    std::cout << "Ostateczna wartość sumy: " << vectorSum << std::endl;
    std::cout << "Ostateczna wartość iloczyny: " << vectorProduct << std::endl;

    return 0;
}