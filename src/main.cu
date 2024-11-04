#include <iostream>
#include <stdio.h>
#include "util.h"
int main() {
    double start_time = get_time();
    std::cout << "Hello, CMake!" << std::endl;
    double end_time = get_time();
    printf("%f\n",end_time-start_time);
    return 0;
}