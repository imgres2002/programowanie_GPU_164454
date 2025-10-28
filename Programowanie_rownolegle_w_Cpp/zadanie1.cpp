#include <iostream>
#include <thread>
#include <mutex>

std::mutex mtx; // mutex do synchronizacji wątków
int counter = 0; // współdzielony zasób

// funkcja przeznaczona do paralelizacji
void incrementCounter(int id, int counter) {
    for (int i = 0; i < 1000; ++i) {
        mtx.lock();
        counter++;
        std::cout << "Wątek " << id << ", counter " << counter << std::endl;
        mtx.unlock();
    }
}

// funkcja przeznaczona do paralelizacji
void decrementCounter(int id, int counter) {
    for (int i = 0; i < 1000; ++i) {
        mtx.lock();
        counter--;
        std::cout << "Wątek " << id << ", counter " << counter << std::endl;
        mtx.unlock();
    }
}

int main() {
    std::cout << "Uruchamianie wątków..." << std::endl;

    // tworzenie dwóch wątków
    std::thread thread1(incrementCounter, 1, counter);
    std::thread thread2(decrementCounter, 2, counter);

    // oczekiwanie na zakończenie wątków
    thread1.join();
    thread2.join();

    std::cout << "Ostateczna wartość counter: " << counter << std::endl;

    return 0;
}