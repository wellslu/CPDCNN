#include <iostream>
#include <gtest/gtest.h>
#include "util.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

TEST(tensor_transformation, reshape_input_tensor) {
    // Set filter size
    int filter_h = 2;
    int filter_w = 2;

    // Create an input tensor [C=1, H=3, W=3] with values from 1 to 9
    torch::Tensor input = torch::arange(1, 1 * 3 * 3 + 1).view({1, 3, 3}).to(torch::kFloat);

    // Extract input dimensions
    int C = input.size(0);  // Number of channels
    int H = input.size(1);  // Height
    int W = input.size(2);  // Width

    // Call the tensor_transformation function
    torch::Tensor output = tensor_transformation(input, filter_h, filter_w);

    // Expected output dimensions
    int H_new = H - filter_h + 1;  
    int W_new = W - filter_w + 1;
    std::vector<int64_t> expected_shape = {H_new, W_new, filter_h, filter_w, C};

    // Test the shape of the output tensor
    EXPECT_EQ(output.sizes().vec(), expected_shape) << "Output shape mismatch.";

    // Expected output tensor values (manually computed patches)
    torch::Tensor expected_output = torch::tensor(
        {
            {{{{1, 2}, {4, 5}}, {{2, 3}, {5, 6}}},  // Patches in (hi, wi)
             {{{4, 5}, {7, 8}}, {{5, 6}, {8, 9}}}},
        },
        torch::kFloat
    );

    // Flatten both tensors for comparison
    torch::Tensor flattened_output = output.flatten();
    torch::Tensor flattened_expected = expected_output.flatten();

    EXPECT_EQ(flattened_output.sizes(), flattened_expected.sizes())
        << "Flattened shape mismatch.";

    // Assert values
    EXPECT_TRUE(torch::allclose(flattened_output, flattened_expected))
        << "Flattened tensor values mismatch.\nExpected:\n"
        << flattened_expected << "\nActual:\n"
        << flattened_output;
}

torch::Tensor create_diagonal_matrix(int rows, int cols) {
        torch::Tensor diag = torch::eye(std::min(rows, cols)).to(torch::kFloat);
        if (rows > cols) {
            return torch::cat({diag, torch::zeros({rows - cols, cols}).to(torch::kFloat)}, 0);
        } else if (cols > rows) {
            return torch::cat({diag, torch::zeros({rows, cols - rows}).to(torch::kFloat)}, 1);
        }
        return diag;
};

TEST(tensor_contraction_sgemv, einsum){
    torch::Tensor test = torch::arange(1, 2*3*4*5 + 1).view({2, 3, 4, 5}).to(torch::kFloat).contiguous();

    std::vector<torch::Tensor> factors = {
        create_diagonal_matrix(7, 6),
        create_diagonal_matrix(3, 6),
        create_diagonal_matrix(4, 6),
        create_diagonal_matrix(5, 1)
        // torch::tensor({1, 0, 0, 0, 0, 0, 1, 0, 0, 0}).reshape({5, 2}).to(torch::kFloat)
        // torch::tensor({1, 0, 0, 0, 0}).to(torch::kFloat)
        // torch::tensor({1, 0, 0, 0, 0}).reshape({5, 1}).to(torch::kFloat)
    };

    // Assign tensor sizes to a variable
    std::vector<int64_t> tensor_sizes = test.sizes().vec();
    int batch_count = factors[3].size(1); // factors[3].size(0)/tensor_sizes[3];
    int three_size = 1;
    for(int i=0; i<3; i++){
        three_size *= tensor_sizes[i];
    }

    torch::Tensor y_3 = torch::zeros({three_size}).to(torch::kFloat); // y initialized with 0

    float *d_A, *d_x3, *d_y3;
    // Allocate device memory
    cudaMalloc((void**)&d_A, three_size * tensor_sizes[3] * sizeof(float));
    cudaMalloc((void**)&d_x3, tensor_sizes[3] * batch_count * sizeof(float));
    cudaMalloc((void**)&d_y3, (three_size) * sizeof(float));

    // Copy data to device memory
    cudaMemcpy(d_A, test.data_ptr<float>(), (three_size) * tensor_sizes[3] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x3, factors[3].data_ptr<float>(), tensor_sizes[3] * batch_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y3, y_3.data_ptr<float>(), (three_size) * sizeof(float), cudaMemcpyHostToDevice);
    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Set parameters for SGEMV
    float alpha = 1.;
    float beta = 1;
    cublasOperation_t trans = CUBLAS_OP_T;
    int lda = tensor_sizes[3];             // Leading dimension of A (number of rows)
    int stride_A = 0;                 // Reuse the same d_A for all batches
    int stride_x = tensor_sizes[3];   // Stride between columns in factors[3]
    int stride_y = 0;        // No stride for d_y3, accumulate into a single vector

    cublasSgemv(
        handle, trans, tensor_sizes[3], three_size,
        &alpha, d_A, lda,
        d_x3, 1,
        &beta, d_y3, 1
    );

    // Allocate host memory to retrieve the data
    float* h_y = new float[(three_size) * batch_count];
    cudaMemcpy(h_y, d_y3, (three_size) * batch_count * sizeof(float), cudaMemcpyDeviceToHost);

    torch::Tensor reshaped = test.view({24, 5}); // Reshape into (24, 5)
    torch::Tensor transposed = reshaped.t(); // Transpose to (5, 24)
    torch::Tensor expected = transposed.slice(0, 0, 1).squeeze(); // Extract first column

    // Perform the comparison
    for (int i = 0; i < expected.size(0); ++i) {
        EXPECT_EQ(h_y[i], expected[i].item<float>()) << "Mismatch at index " << i;
    }

    // Clean up memory
    delete[] h_y;
    cudaFree(d_A);
    cudaFree(d_x3);
    cudaFree(d_y3);
    cublasDestroy(handle);
}

TEST(tensor_contraction_sgemm, einsum){
    torch::Tensor test = torch::arange(1, 2*3*4*5 + 1).view({2, 3, 4, 5}).to(torch::kFloat).contiguous();

    std::vector<torch::Tensor> factors = {
        create_diagonal_matrix(7, 6),
        create_diagonal_matrix(3, 6),
        create_diagonal_matrix(4, 6),
        create_diagonal_matrix(5, 6),
    };

    // Assign tensor sizes to a variable
    std::vector<int64_t> tensor_sizes = test.sizes().vec();
    int batch_count = factors[3].size(1);
    int three_size = 1;
    for(int i=0; i<3; i++){
        three_size *= tensor_sizes[i];
    }

    torch::Tensor y_3 = torch::zeros({three_size * batch_count}).to(torch::kFloat); // y initialized with 0
    torch::Tensor ones = torch::ones({batch_count}).to(torch::kFloat); // one initialized with 1
    // torch::Tensor y_2 = torch::zeros({tensor_sizes[0]*tensor_sizes[1]}).to(torch::kFloat); // y initialized with 0
    // torch::Tensor y_1 = torch::zeros({tensor_sizes[0]}).to(torch::kFloat); // y initialized with 0
    // Device pointers
    // float *d_A, *d_x1, *d_x2, *d_x3, *d_y;
    float *d_A, *d_x3, *d_y3, *d_row_sums, *d_ones;
    // Allocate device memory
    cudaMalloc((void**)&d_A, three_size * tensor_sizes[3] * sizeof(float));
    cudaMalloc((void**)&d_x3, tensor_sizes[3] * batch_count * sizeof(float));
    // cudaMalloc((void**)&d_x2, sizeof(factors[2]));//factors[2]
    // cudaMalloc((void**)&d_x1, sizeof(factors[1]));//factors[1]
    // cudaMalloc((void**)&d_yy, sizeof(factors[0]));//factors[0]
    cudaMalloc((void**)&d_y3, (three_size) * batch_count * sizeof(float));
    cudaMalloc((void**)&d_ones, batch_count * sizeof(float));
    cudaMalloc((void**)&d_row_sums, (three_size) * sizeof(float));
    // cudaMalloc((void**)&d_y2, sizeof(y_2));
    // cudaMalloc((void**)&d_y1, sizeof(y_1));

    // Copy data to device memory
    cudaMemcpy(d_A, test.data_ptr<float>(), (three_size) * tensor_sizes[3] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x3, factors[3].data_ptr<float>(), tensor_sizes[3] * batch_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y3, y_3.data_ptr<float>(), (three_size) * batch_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ones, ones.data_ptr<float>(), batch_count * sizeof(float), cudaMemcpyHostToDevice);
    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Set parameters for SGEMV
    float alpha = 1.;
    float beta = 1;
    cublasOperation_t trans = CUBLAS_OP_T;
    int lda = tensor_sizes[3];             // Leading dimension of A (number of rows)

    cublasSgemm(
        handle,
        trans, trans,
        three_size, batch_count, tensor_sizes[3],
        &alpha,
        d_A, tensor_sizes[3],//(5x24)^T
        d_x3, batch_count,//(6x5)^T
        &beta,
        d_y3, three_size//(24x6) column-major
    );

    // Allocate host memory to retrieve the data
    float* h_y = new float[(three_size) * batch_count];
    cudaMemcpy(h_y, d_y3, (three_size) * batch_count * sizeof(float), cudaMemcpyDeviceToHost);

    // // Print the result vector y
    // std::cout << "Result vector (y) first column:" << std::endl;
    // for (int i = 0; i < (three_size) * batch_count; i++) {
    //     std::cout << h_y[i] << std::endl;
    // }
    torch::Tensor reshaped = test.view({three_size, tensor_sizes[3]}); // Reshape into (24, 5)
    torch::Tensor transposed = reshaped.t(); // Transpose to (5, 24)
    torch::Tensor flattened = transposed.contiguous().flatten();
    // Perform the comparison
    int min = (batch_count < tensor_sizes[3]) ? batch_count : tensor_sizes[3];
    for (int i = 0; i < (three_size) * min; ++i) {
        EXPECT_EQ(h_y[i], flattened[i].item<float>()) << "Mismatch at index " << i;
    }

    cublasSgemv(
        handle,
        CUBLAS_OP_N,            // No transpose for d_y3
        three_size, batch_count, // Rows = 24, Columns = 6
        &alpha,                 // Scaling factor (1.0)
        d_y3, three_size,       // Matrix d_y3
        d_ones, 1,              // Vector d_ones
        &beta,                  // Scaling factor for existing d_row_sums (0.0)
        d_row_sums, 1           // Output: row sums
    );

    // Allocate host memory to retrieve the data
    float* h_row_sums = new float[three_size];
    cudaMemcpy(h_row_sums, d_row_sums, three_size * sizeof(float), cudaMemcpyDeviceToHost);
    torch::Tensor result = reshaped.sum(1); // Result is a (24,) tensor
    auto expected_row_sums = result.data_ptr<float>(); // Access raw data
     for (int i = 0; i < three_size; i++) {
        EXPECT_EQ(h_row_sums[i], expected_row_sums[i]) << "Mismatch at index " << i;
    }

    // Clean up memory
    delete[] h_y, h_row_sums;
    cudaFree(d_A);
    cudaFree(d_x3);
    cudaFree(d_y3);
    cudaFree(d_ones);
    cudaFree(d_row_sums);
    cublasDestroy(handle);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}