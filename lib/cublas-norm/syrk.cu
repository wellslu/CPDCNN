#include "cublas-norm/syrk.h"
#include <iostream>

// Matrix constructor implementation
Matrix::Matrix(int A_N, double A_coff, int C_N, double C_coff)
    : A_N(A_N), C_N(C_N), alpha(A_coff), beta(C_coff) {
    A = new double[A_N * A_N]();
    for (int i = 0; i < A_N; ++i) {
        A[i * A_N + i] = 1.0; // Set diagonal elements to 1
    }
    C = new double[C_N * C_N]();
    for (int i = 0; i < C_N; ++i) {
        C[i * C_N + i] = 1.0; // Set diagonal elements to 1
    }
}

// Matrix destructor implementation
Matrix::~Matrix() {
    delete[] A;
    delete[] C;
}

// Accessor methods
double* Matrix::getA() const { return A; }
double Matrix::getAlpha() const { return alpha; }
double* Matrix::getC() const { return C; }
double Matrix::getBeta() const { return beta; }

// Info class constructor implementations
Info::Info(int A_N, double A_coff, int C_N, double C_coff, int iter, double val, double t)
    : matrix(A_N, A_coff, C_N, C_coff), iteration(iter), value(val), time(t) {}

Info::Info() : matrix(1, 1.0, 1, 1.0), iteration(0), value(0.0), time(0.0) {}

// Implementation of the it_syrk function
void it_syrk(Info* result) {
    int N = result->matrix.A_N;
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
                                            N,
                                            N,
                                            &alpha,
                                            d_A,
                                            N,
                                            &beta,
                                            d_C,
                                            N);

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
    result->time = elapsedTime / 1000.0;

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