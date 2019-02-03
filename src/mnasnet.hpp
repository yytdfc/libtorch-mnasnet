#include <torch/torch.h>
namespace mnasnet{
using namespace std;

//in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
auto Conv2d(size_t in, size_t out, size_t kernel, size_t stride=1, size_t padding=0, size_t dilation=1, size_t groups=1, bool bias=true){
  auto opt = torch::nn::Conv2dOptions(in, out, kernel);
  opt.stride(stride);
  opt.padding(padding);
  opt.dilation(dilation);
  opt.groups(groups);
  opt.with_bias(bias);
  return torch::nn::Conv2d(opt);
}

auto Conv3x3(size_t in, size_t out, size_t stride=1, size_t dilation=1, size_t groups=1){
  return torch::nn::Sequential(
    Conv2d(in, out, 3, stride, 1, dilation, groups, false),
    torch::nn::BatchNorm(out),
    torch::nn::Functional(torch::relu)
  );
}

auto Conv1x1(size_t in, size_t out, size_t stride=1, size_t dilation=1, size_t groups=1){
  return torch::nn::Sequential(
    Conv2d(in, out, 1, stride, 0, dilation, groups, false),
    torch::nn::BatchNorm(out),
    torch::nn::Functional(torch::relu)
  );
}

auto SepConv3x3(size_t in, size_t out, size_t stride=1, size_t dilation=1){
  return torch::nn::Sequential(
    // dw
    Conv2d(in, in, 3, stride, 1, dilation, in, false),
    torch::nn::BatchNorm(in),
    torch::nn::Functional(torch::relu),
    // pw-linear
    Conv2d(in, out, 1, stride, 0, dilation, 1, false),
    torch::nn::BatchNorm(out)
  );
}

// Define a new Module.
struct InvertedResidual : torch::nn::Module {

  torch::nn::Sequential conv{nullptr};
  bool is_res = false;

  InvertedResidual(size_t in, size_t out, size_t kernel, size_t stride, size_t expand_ratio, size_t dilation=1) {
    auto mid = in * expand_ratio;
    if(in==out && stride==1)
      is_res = true;
    conv = register_module("conv", torch::nn::Sequential(
          // pw
          Conv2d(in, mid, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0, /*dilation*/ 1, /*groups*/ 1, false),
          torch::nn::BatchNorm(mid),
          torch::nn::Functional(torch::relu),
          // dw
          Conv2d(mid, mid, kernel, stride, /*padding*/ /*padding*/ (kernel / 2), dilation, /*groups*/ mid, false),
          torch::nn::BatchNorm(mid),
          torch::nn::Functional(torch::relu),
          // pw-linear
          Conv2d(mid, out, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0, dilation, /*groups*/ 1, false),
          torch::nn::BatchNorm(out)
          ));
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x) {
    if(is_res)
      return x + conv->forward(x);
    else
      return conv->forward(x);
  }
};

struct MnasNet : torch::nn::Module {
  size_t config[6][5] = {
    // t, c, n, s, k
    {3, 24,  3, 2, 3},  // -> 56x56
    {3, 40,  3, 2, 5},  // -> 28x28
    {6, 80,  3, 2, 5},  // -> 14x14
    {6, 96,  2, 1, 3},  // -> 14x14
    {6, 192, 4, 2, 5},  // -> 7x7
    {6, 320, 1, 1, 3},  // -> 7x7
  };
  vector<torch::nn::Sequential> layers;

  MnasNet(size_t n_class=1000, float width_mult=1.0) {
    size_t input_channel = 32 * width_mult;
    layers.emplace_back(register_module("layer0", Conv3x3(/*in*/ 3, /*out*/ input_channel, /*stride*/ 2)));
    layers.emplace_back(register_module("layer1", SepConv3x3(/*in*/ input_channel, /*out*/ 16)));
    input_channel = 16;
    for(size_t i=0; i!=6; ++i){
      size_t output_channel = config[i][1];
      size_t stride = config[i][3];
      torch::nn::Sequential layer;
      for(size_t j=0; j!=config[i][2]; ++j){
        shared_ptr<InvertedResidual> l(
            new InvertedResidual(
              input_channel,
              output_channel,
              /*kernel*/ config[i][4],
              /*stride*/ stride,
              /*expand_ratio*/ config[i][0]
              ));
        layer->push_back(l);
        input_channel = output_channel;
        stride = 1;
      }
      layers.emplace_back(move(register_module("layer" + to_string(i+2), layer)));
    }

  }

  torch::Tensor forward(torch::Tensor x) {
    for(auto& layer: layers){
      x = layer->forward(x);
    }
    return x;
    // auto l0 = layers[0]->forward(x);
    // auto l1 = layers[1]->forward(l0);
    // auto l2 = layers[2]->forward(l1);
    // auto l3 = layers[3]->forward(l2);
    // auto l4 = layers[4]->forward(l3);
    // auto l5 = layers[5]->forward(l4);
    // auto l6 = layers[6]->forward(l5);
    // auto l7 = layers[7]->forward(l6);
    // return l7;
  }
};
}

