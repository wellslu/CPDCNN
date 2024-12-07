#include <torch/torch.h>
double get_time();
torch::Tensor tensor_transformation(torch::Tensor tensor, int filter_h, int filter_w);

class Cuutil {
public:
    torch::Tensor tmp1;
    torch::Tensor tmp2;
    torch::Tensor ones;
    torch::Tensor output;

    Cuutil();
    torch::Tensor tensorcontraction(torch::Tensor &input, std::vector<torch::Tensor>& factors);
};