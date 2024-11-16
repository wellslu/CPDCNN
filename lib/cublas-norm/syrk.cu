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


// If want to add other module extension, this is as template(modify setup.py), 
// Include <torch/extension.h> and register the function only if compiling with setup.py
// #ifdef BUILD_WITH_PYTORCH
// #include <pybind11/numpy.h>
// #include <torch/extension.h>
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     pybind11::class_<Matrix>(m, "Matrix")
//         .def(pybind11::init<int, double, int, double>(),
//              pybind11::arg("A_N"), pybind11::arg("A_coff"),
//              pybind11::arg("C_N"), pybind11::arg("C_coff"))
//         .def("getA", [](const Matrix &matrix) {
//             // Use the public member `A_N` directly
//             return pybind11::array_t<double>({matrix.A_N, matrix.A_N}, matrix.getA());
//         })
//         .def("getC", [](const Matrix &matrix) {
//             // Use the public member `C_N` directly
//             return pybind11::array_t<double>({matrix.C_N, matrix.C_N}, matrix.getC());
//         })
//         .def("getAlpha", &Matrix::getAlpha)
//         .def("getBeta", &Matrix::getBeta)
//         .def_readonly("A_N", &Matrix::A_N)
//         .def_readonly("C_N", &Matrix::C_N);

//     pybind11::class_<Info>(m, "Info")
//         .def(pybind11::init<int, double, int, double, int, double, double>(),
//              pybind11::arg("A_N"), pybind11::arg("A_coff"),
//              pybind11::arg("C_N"), pybind11::arg("C_coff"),
//              pybind11::arg("iteration"), pybind11::arg("value"),
//              pybind11::arg("time"))
//         .def(pybind11::init<>()) // Default constructor
//         .def_property_readonly("matrix", [](const Info &info) {
//             return &info.matrix;
//         }, pybind11::return_value_policy::reference)
//         .def_readwrite("iteration", &Info::iteration)
//         .def_readwrite("value", &Info::value)
//         .def_readwrite("time", &Info::time);
// }
// #endif