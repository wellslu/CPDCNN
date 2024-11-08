#include <iostream>
#include <gtest/gtest.h>
#include "cublas-norm/syrk.h"

TEST(syrkParallel, Speedup) {
    int N = 1024;
    double alpha = 0.001;
    int iterations = 1000;
    double expected_diagonal_value = 1.0 + iterations * alpha; // Expected value on the diagonal
    double expected_trace = expected_diagonal_value * N;        // Total expected trace

    Info result(N, alpha, N, 1.0, iterations, 0.0, 0.0);
    it_syrk(&result);

    // Use EXPECT_NEAR due to floating-point precision
    EXPECT_NEAR(result.value, expected_trace, 1e-2 * expected_trace); // Allow 1% tolerance
    std::cout << "Computation time: " << result.time << " seconds" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}