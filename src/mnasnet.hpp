#include <torch/torch.h>
namespace mnasnet{
using namespace std;

//in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
auto Conv2d(size_t in, size_t out, size_t kernel, size_t stride=1, size_t padding=0, size_t dilation=1, size_t groups=1, bool bias=true){
  auto opt = torch::nn::Conv2dOptions(in, out, 3);
  opt.stride(stride);
  opt.padding(padding);
  opt.dilation(dilation);
  opt.groups(groups);
  opt.with_bias(bias);
  return torch::nn::Conv2d(opt);
}


auto Conv3x3(size_t in, size_t out, size_t kernel, size_t stride=1, size_t padding=0, size_t dilation=1, size_t groups=1, bool bias=true){
  return torch::nn::Sequential(
    Conv2d(in, out, 3, stride, 1, dilation, groups, bias),
    torch::nn::BatchNorm(out),
    torch::nn::Functional(torch::relu)
  );
  // return seq;
}




// Define a new Module.
struct Net : torch::nn::Module {
  Net() {
    // Construct and register two Linear submodules.
    fc1 = register_module("fc1", torch::nn::Linear(64, 64));
    fc2 = register_module("fc2", torch::nn::Linear(64, 32));
    fc3 = register_module("fc3", torch::nn::Linear(32, 10));
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x) {
    // Use one of many tensor manipulation functions.
    x = torch::relu(fc1->forward(x));
    x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
    x = torch::relu(fc2->forward(x));
    x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
    return x;
  }

  // Use one of many "standard library" modules.
  // torch::nn::Conv2d conv1;
  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};
}

