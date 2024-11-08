#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <gtest/gtest.h>

class Matrix {
public:
    int A_N, C_N; // Sizes of matrices A and C

private:
    double* A;    // Pointer to hold the A matrix
    double alpha;
    double* C;    // Pointer to hold the C matrix
    double beta;

public:
    // Constructor to initialize matrices A and C as diagonal matrices with element 1
    Matrix(int A_N, double A_coff, int C_N, double C_coff)
        : A_N(A_N), C_N(C_N), alpha(A_coff), beta(C_coff)
    {
        // Allocate memory for A (A_N x A_N) as a contiguous block
        A = new double[A_N * A_N]();
        for (int i = 0; i < A_N; ++i) {
            A[i * A_N + i] = 1.0; // Set diagonal elements to 1
        }

        // Allocate memory for C (C_N x C_N) as a contiguous block
        C = new double[C_N * C_N]();
        for (int i = 0; i < C_N; ++i) {
            C[i * C_N + i] = 1.0; // Set diagonal elements to 1
        }
    }

    // Destructor to free allocated memory
    ~Matrix() {
        delete[] A;
        delete[] C;
    }

    // Accessor methods
    double* getA() const { return A; }
    double getAlpha() const { return alpha; }
    double* getC() const { return C; }
    double getBeta() const { return beta; }
};

class Info {
public:
    Matrix matrix;
    int iteration;
    double value;
    double time;

    // Constructor to initialize Info object with Matrix and other attributes
    Info(int A_N, double A_coff, int C_N, double C_coff, int iter, double val, double t)
        : matrix(A_N, A_coff, C_N, C_coff), iteration(iter), value(val), time(t)
    {
    }

    // Default constructor with no arguments
    Info() : matrix(1, 1.0, 1, 1.0), iteration(0), value(0.0), time(0.0) {
    }
};

void it_syrk(Info* result) {
    // Get matrix sizes and coefficients from the result structure
    int N = result->matrix.A_N; // Assuming A_N and C_N are the same
    double alpha = result->matrix.getAlpha();
    double beta = result->matrix.getBeta();
    int iterations = result->iteration;

    // Allocate memory on the device for matrices d_A and d_C
    double* d_A;
    double* d_C;
    cudaMalloc((void**)&d_A, N * N * sizeof(double));
    cudaMalloc((void**)&d_C, N * N * sizeof(double));

    // Copy matrix A from host to device
    cudaMemcpy(d_A, result->matrix.getA(), N * N * sizeof(double), cudaMemcpyHostToDevice);

    // Copy matrix C from host to device
    cudaMemcpy(d_C, result->matrix.getC(), N * N * sizeof(double), cudaMemcpyHostToDevice);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);
    for (int iter = 0; iter < iterations; ++iter) {
        cublasStatus_t status = cublasDsyrk(handle,
                                            CUBLAS_FILL_MODE_UPPER,
                                            CUBLAS_OP_N,
                                            N,        // n
                                            N,        // k
                                            &alpha,   // alpha
                                            d_A,      // A
                                            N,        // lda
                                            &beta,    // beta
                                            d_C,      // C
                                            N);       // ldc

        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cuBLAS SYRK failed at iteration " << iter << std::endl;
            cublasDestroy(handle);
            cudaFree(d_A);
            cudaFree(d_C);
            return;
        }
    }
    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time in milliseconds
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    result->time = elapsedTime / 1000.0;  // Convert to seconds

    // Copy result matrix C back to the host
    cudaMemcpy(result->matrix.getC(), d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    // Compute the trace of matrix C (sum of diagonal elements)
    double* C = result->matrix.getC();
    double trace = 0.0;
    for (int i = 0; i < N; ++i) {
        trace += C[i * N + i];
    }
    result->value = trace;

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_C);
}

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