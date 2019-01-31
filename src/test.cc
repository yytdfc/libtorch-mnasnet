#include <torch/torch.h>
#include <iostream>
#include "mnasnet.hpp"
using namespace mnasnet;

int main() {
  auto conv3x3 = Conv3x3(1, 1, 1);
  cout << conv3x3 <<endl;

  // Create a new Net.

  // auto net = std::make_shared<Net>();

  torch::Tensor tensor = torch::rand({1, 1, 8, 8});

  std::cout << conv3x3->forward(tensor) << std::endl;
}
