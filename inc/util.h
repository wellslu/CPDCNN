#include <torch/torch.h>
double get_time();
torch::Tensor tensor_transformation(torch::Tensor tensor, int filter_h, int filter_w);
// torch::Tensor tensorcontraction(torch::Tensor input, std::vector<torch::Tensor>& factors);

class Cuutil {
public:
    torch::Tensor tmp1; // Raw CUDA memory
    torch::Tensor tmp2; // Raw CUDA memory
    torch::Tensor ones; // PyTorch Tensor
    torch::Tensor output; // PyTorch Tensor

    Cuutil();
    ~Cuutil();
    torch::Tensor tensorcontraction(torch::Tensor &input, std::vector<torch::Tensor>& factors);
};