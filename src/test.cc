#include <torch/torch.h>
#include <iostream>
#include "mnasnet.hpp"
using namespace mnasnet;

int main() {
  // auto conv3x3 = Conv3x3(3, 1);
  // auto conv3x3 = Conv1x1(3, 1);
  // auto conv3x3 = SepConv3x3(3, 1);
  // auto conv3x3 = InvertedResidual([>in*/ 3, /*out*/ 1, /*kernel*/ 1, /*stride*/ 1, /*expand_ratio<] 6);
  auto conv3x3 = MnasNet(10);
  cout << conv3x3 <<endl;

  // Create a new Net.

  // auto net = std::make_shared<Net>();

  torch::Tensor tensor = torch::rand({1, 3, 28, 28});

  for(auto& l : conv3x3.features(tensor))
    std::cout << l.sizes() << std::endl;
  std::cout << conv3x3.forward(tensor).sizes() << std::endl;
}
