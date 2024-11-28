#include <torch/torch.h>
double get_time();
torch::Tensor tensor_transformation(torch::Tensor tensor, int filter_h, int filter_w);