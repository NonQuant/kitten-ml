#pragma once

#include "ops.hpp"
#include <cmath>

namespace kittenml::nn {

class Linear {
public:
  TensorPtr W;
  TensorPtr b;

  Linear(int in_features, int out_features)
      : W(std::make_shared<Tensor>(
            Eigen::MatrixXd::Random(in_features, out_features) /
            std::sqrt(static_cast<double>(in_features)))),
        b(std::make_shared<Tensor>(Eigen::MatrixXd::Zero(1, out_features))) {}

  TensorPtr forward(const TensorPtr &x) const { return matmul(x, W) + b; }
};

} // namespace kittenml::nn
