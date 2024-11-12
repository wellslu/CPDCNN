#include <sys/time.h>

double get_time() {
   struct timeval t;
   gettimeofday(&t, NULL);
   return t.tv_sec + t.tv_usec / 1000000.0;
}

// Include <torch/extension.h> and register the function only if compiling with setup.py
#ifdef BUILD_WITH_PYTORCH
#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_time", &get_time, "Get Current Time");
}
#endif