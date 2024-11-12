#include <iostream>
#include "util.h"
int main() {
    double start_time = get_time();
    std::cout << "Hello, CMake!" << std::endl;
    double end_time = get_time();
    std::cout << "Time interval: " << end_time - start_time << std::endl;
    return 0;
}