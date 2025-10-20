#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cmath>

std::queue<int> buffer;            // Kolejka współdzielona
std::mutex mtx;                    // Mutex do synchronizacji
std::condition_variable cv;        // Zmienna warunkowa
bool done = false;                 // Flaga zakończenia produkcji

const int MAX_ITEMS = 10;          // Liczba elementów do wyprodukowania

// Funkcja producenta
void producer() {
    for (int i = 1; i <= MAX_ITEMS; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Symulacja produkcji

        std::unique_lock<std::mutex> lock(mtx);  // Blokowanie mutexa
        int randomNum = rand() % 11;
        buffer.push(randomNum);
        std::cout << "Producent: Wyprodukowano " << randomNum << std::endl;

        cv.notify_one(); // Powiadomienie konsumenta
    }

    // Ustawienie flagi zakończenia i powiadomienie konsumenta
    std::unique_lock<std::mutex> lock(mtx);
    done = true;
    cv.notify_one();
}

// Funkcja konsumenta
void consumer() {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [] { return !buffer.empty() || done; }); // Oczekiwanie na dane

        if (buffer.empty() && done) {
            break; // Jeśli produkcja zakończona i kolejka pusta -> zakończ działanie
        }

        int item = buffer.front();
        buffer.pop();
        int randomPowerNum = rand() % 5;
        int numToPower = pow(item, randomPowerNum);
        std::cout << "Konsument: Przetwarzanie liczby " << item << std::endl;
        std::cout << "Konsument: Pseudolosowa potęga " << randomPowerNum << std::endl;
        std::cout << "Konsument: Liczba podniesiona do pseudolosowej potęgi " << numToPower << std::endl;
    }
}

int main() {
    std::thread producerThread(producer);
    std::thread consumerThread(consumer);

    producerThread.join();
    consumerThread.join();

    std::cout << "Zakończono produkcję i konsumpcję." << std::endl;
    return 0;
}
