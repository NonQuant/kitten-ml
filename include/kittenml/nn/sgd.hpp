#pragma once

#include <kittenml/tensor.hpp>

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kittenml::nn {

class SGD {
public:
  SGD(std::vector<TensorPtr> parameters, double learning_rate)
      : parameters_(std::move(parameters)), learning_rate_(learning_rate) {
    if (!std::isfinite(learning_rate_) || learning_rate_ <= 0.0)
      throw std::invalid_argument("SGD learning rate must be positive");

    for (const auto &parameter : parameters_) {
      if (!parameter)
        throw std::invalid_argument("SGD parameters must not be null");
    }
  }

  void step() {
    for (const auto &parameter : parameters_) {
      parameter->data -= learning_rate_ * parameter->grad;
      parameter->grad.setZero();
    }
  }

  void zero_grad() {
    for (const auto &parameter : parameters_)
      parameter->grad.setZero();
  }

private:
  std::vector<TensorPtr> parameters_;
  double learning_rate_;
};

} // namespace kittenml::nn
