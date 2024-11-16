#include <sys/time.h>
#include "cublas-norm/syrk.h"
double get_time() {
   struct timeval t;
   gettimeofday(&t, NULL);
   return t.tv_sec + t.tv_usec / 1000000.0;
}

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
}
#endif