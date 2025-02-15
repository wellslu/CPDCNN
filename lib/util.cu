#include <sys/time.h>
#include "util.h"
#include "cublas-norm/syrk.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

double get_time() {
   struct timeval t;
   gettimeofday(&t, NULL);
   return t.tv_sec + t.tv_usec / 1000000.0;
}

Cuutil::Cuutil() {
    tmp1 = torch::zeros({500000}, torch::device(torch::kCUDA).dtype(torch::kFloat));
    tmp2 = torch::zeros({500000}, torch::device(torch::kCUDA).dtype(torch::kFloat));
    ones = torch::ones({10}, torch::device(torch::kCUDA).dtype(torch::kFloat));
    output = torch::zeros({500000}, torch::device(torch::kCUDA).dtype(torch::kFloat));
}

torch::Tensor tensor_transformation(torch::Tensor tensor, int filter_h, int filter_w) {
    // Ensure the tensor has at least 3 dimensions: [C, H, W]
    TORCH_CHECK(tensor.dim() == 3, "Input tensor must have 3 dimensions: [C, H, W]");

    // Extract dimensions from the tensor
    int C = tensor.size(0);  // Number of channels
    int H = tensor.size(1);  // Height of the tensor
    int W = tensor.size(2);  // Width of the tensor

    // Calculate new dimensions
    int H_new = H - filter_h + 1;
    int W_new = W - filter_w + 1;

    // Ensure the filter dimensions are valid
    TORCH_CHECK(H_new > 0 && W_new > 0, "Filter size is larger than input dimensions");

    // Unfold height and width
    auto unfolded = tensor.unfold(1, filter_h, 1).unfold(2, filter_w, 1);  // [C, H_new, filter_h, W_new, filter_w]

    // Permute dimensions to [H_new, W_new, filter_h, filter_w, C]
    auto reshaped_tensor = unfolded.permute({1, 2, 3, 4, 0});  // Rearrange to match manual calculation

    return reshaped_tensor;
}

torch::Tensor Cuutil::tensorcontraction(
    torch::Tensor &input,
    std::vector<torch::Tensor>& factors
    ){
    float *d_input, *d_factor3, *d_factor2, *d_factor1, *d_factor0;
    float *d_intrmdt3, *d_intrmdt2, *d_ones;
    float *d_y3, *d_y2, *d_y1, *d_y0, *d_output;
    cublasHandle_t handle;
    std::vector<int64_t> input_sizes = input.sizes().vec();

    int64_t inpt_total = 1; // Initialize the product
    for (const auto& size : input_sizes) {
        inpt_total *= size; // Multiply each element
    }

    int cpdrk = factors[3].size(1);
    ones = torch::ones({cpdrk}, torch::device(torch::kCUDA).dtype(torch::kFloat));

    float alpha = 1.;
    float beta = 1;
    cublasOperation_t trans = CUBLAS_OP_T;
    /* mode-6(mode-5th) tensor contraction */
    // Batch x H_new x W_new x H_filter x W_filter x C [0,1,2,3,4,5]
    tmp1 = torch::zeros({(input_sizes[0]*input_sizes[1]*input_sizes[2]*input_sizes[3]*input_sizes[4]) * cpdrk}, torch::device(torch::kCUDA).dtype(torch::kFloat));
    //sgemm
    cublasCreate(&handle);
    cublasSgemm(
        handle,
        trans, trans,
        (input_sizes[0]*input_sizes[1]*input_sizes[2]*input_sizes[3]*input_sizes[4]), cpdrk, input_sizes[5],
        &alpha,
        input.data_ptr<float>(), input_sizes[5],//(5x24)^T
        factors[3].data_ptr<float>(), cpdrk,//(6x5)^T
        &beta,
        tmp1.data_ptr<float>(), (input_sizes[0]*input_sizes[1]*input_sizes[2]*input_sizes[3]*input_sizes[4])//(24x6) column-major
    );

    factors[2] = factors[2].t().contiguous();
    /* mode-4th tensor contraction */
    tmp2 = torch::zeros({(input_sizes[0]*input_sizes[1]*input_sizes[2]*input_sizes[3]) * cpdrk}, torch::device(torch::kCUDA).dtype(torch::kFloat));
    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,                // transa, transb
        input_sizes[0]*input_sizes[1]*input_sizes[2]*input_sizes[3], 1, input_sizes[4],                  // m, n, k
        &alpha,                                 // alpha
        tmp1.data_ptr<float>(), input_sizes[4],             // A, lda
        input_sizes[0]*input_sizes[1]*input_sizes[2]*input_sizes[3]*input_sizes[4], // strideA
        factors[2].data_ptr<float>(), input_sizes[4],              // B, ldb
        input_sizes[4],                         // strideB
        &beta,                                  // beta
        tmp2.data_ptr<float>(), input_sizes[0]*input_sizes[1]*input_sizes[2]*input_sizes[3], // C, ldc
        input_sizes[0]*input_sizes[1]*input_sizes[2]*input_sizes[3],       // strideC
        cpdrk                                  // batchCount
    );

    /* mode-3th tensor contraction */
    factors[1] = factors[1].t().contiguous();
    tmp1 = torch::zeros({(input_sizes[0]*input_sizes[1]*input_sizes[2]) * cpdrk}, torch::device(torch::kCUDA).dtype(torch::kFloat));
    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,                // transa, transb
        input_sizes[0]*input_sizes[1]*input_sizes[2], 1, input_sizes[3],                  // m, n, k
        &alpha,                                 // alpha
        tmp2.data_ptr<float>(), input_sizes[3],             // A, lda
        input_sizes[0]*input_sizes[1]*input_sizes[2]*input_sizes[3], // strideA
        factors[1].data_ptr<float>(), input_sizes[3],              // B, ldb
        input_sizes[3],                         // strideB
        &beta,                                  // beta
        tmp1.data_ptr<float>(), input_sizes[0]*input_sizes[1]*input_sizes[2], // C, ldc
        input_sizes[0]*input_sizes[1]*input_sizes[2],       // strideC
        cpdrk                                  // batchCount
    );

    // mode-2th outer product
    factors[0] = factors[0].t().contiguous();
    tmp2 = torch::zeros({(input_sizes[0]*input_sizes[1]*input_sizes[2]*factors[0].size(1)) * cpdrk}, torch::device(torch::kCUDA).dtype(torch::kFloat));
    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,                // transa, transb
        input_sizes[0]*input_sizes[1]*input_sizes[2], factors[0].size(1), 1,                  // m, n, k
        &alpha,                                 // alpha
        tmp1.data_ptr<float>(), input_sizes[0]*input_sizes[1]*input_sizes[2],             // A, lda
        input_sizes[0]*input_sizes[1]*input_sizes[2], // strideA
        factors[0].data_ptr<float>(), factors[0].size(1),              // B, ldb
        factors[0].size(1),                         // strideB
        &beta,                                  // beta
        tmp2.data_ptr<float>(), input_sizes[0]*input_sizes[1]*input_sizes[2], // C, ldc
        input_sizes[0]*input_sizes[1]*input_sizes[2]*factors[0].size(1),       // strideC
        cpdrk                                  // batchCount
    );

    output = torch::zeros({input_sizes[0] * input_sizes[1] * input_sizes[2] * factors[0].size(1)}, torch::device(torch::kCUDA).dtype(torch::kFloat));
    cublasSgemv(
        handle,
        CUBLAS_OP_N,
        input_sizes[0]*input_sizes[1]*input_sizes[2]*factors[0].size(1), cpdrk,
        &alpha,
        tmp2.data_ptr<float>(), input_sizes[0]*input_sizes[1]*input_sizes[2]*factors[0].size(1),
        ones.data_ptr<float>(), 1,
        &beta,
        output.data_ptr<float>(), 1
    );

    cublasDestroy(handle);

    output = output.reshape({factors[0].size(1), input_sizes[0]*input_sizes[1]*input_sizes[2]});
    return output;
}

// torch::Tensor taconvfoward(torch::Tensor input, std::vector<torch::Tensor> factors, int filter_h, int filter_w, int rank) {
//     // generate 5-way reshape
//     auto reshape = tensor_transformation(input, filter_h, filter_w);
//     // paralell with rank and add those result
//     auto output += tensorcontraction<<>>(reshape, factors);
//     return output;
// }

// Include <torch/extension.h> and register the function only if compiling with setup.py
#ifdef BUILD_WITH_PYTORCH
#include <pybind11/numpy.h>
#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_time", &get_time, "Get Current Time");
    pybind11::class_<Matrix>(m, "Matrix")
        .def(pybind11::init<int, double, int, double>(),
             pybind11::arg("A_N"), pybind11::arg("A_coff"),
             pybind11::arg("C_N"), pybind11::arg("C_coff"))
        .def("getA", [](const Matrix &matrix) {
            // Use the public member `A_N` directly
            return pybind11::array_t<double>({matrix.A_N, matrix.A_N}, matrix.getA());
        })
        .def("getC", [](const Matrix &matrix) {
            // Use the public member `C_N` directly
            return pybind11::array_t<double>({matrix.C_N, matrix.C_N}, matrix.getC());
        })
        .def("getAlpha", &Matrix::getAlpha)
        .def("getBeta", &Matrix::getBeta)
        .def_readonly("A_N", &Matrix::A_N)
        .def_readonly("C_N", &Matrix::C_N);

    pybind11::class_<Info>(m, "Info")
        .def(pybind11::init<int, double, int, double, int, double, double>(),
             pybind11::arg("A_N"), pybind11::arg("A_coff"),
             pybind11::arg("C_N"), pybind11::arg("C_coff"),
             pybind11::arg("iteration"), pybind11::arg("value"),
             pybind11::arg("time"))
        .def(pybind11::init<>()) // Default constructor
        .def_property_readonly("matrix", [](const Info &info) {
            return &info.matrix;
        }, pybind11::return_value_policy::reference)
        .def_readwrite("iteration", &Info::iteration)
        .def_readwrite("value", &Info::value)
        .def_readwrite("time", &Info::time);

    //  m.def("tensorcontraction", &tensorcontraction,
    //       "Perform tensor contraction operation",
    //       pybind11::arg("input"), pybind11::arg("factors"));

    pybind11::class_<Cuutil>(m, "Cuutil")
        .def(pybind11::init<>()) // Expose the constructor
        .def("tensorcontraction", &Cuutil::tensorcontraction, pybind11::arg("input"), pybind11::arg("factors"),
             "Performs tensor contraction")
        .def("__repr__", [](const Cuutil &c) {
            return "<Cuutil instance>";
        });
}
#endif