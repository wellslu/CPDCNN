#ifndef SYRK_H
#define SYRK_H

#include <cublas_v2.h>
#include <cuda_runtime.h>

// Declaration of Matrix class
class Matrix {
public:
    int A_N, C_N; // Sizes of matrices A and C

private:
    double* A;    // Pointer to hold the A matrix
    double alpha;
    double* C;    // Pointer to hold the C matrix
    double beta;

public:
    Matrix(int A_N, double A_coff, int C_N, double C_coff);  // Constructor
    ~Matrix();                                               // Destructor

    // Accessor methods
    double* getA() const;
    double getAlpha() const;
    double* getC() const;
    double getBeta() const;
};

// Declaration of Info class
class Info {
public:
    Matrix matrix;
    int iteration;
    double value;
    double time;

    Info(int A_N, double A_coff, int C_N, double C_coff, int iter, double val, double t);
    Info();
};

// Declaration of it_syrk function
void it_syrk(Info* result);

#endif // SYRK_H