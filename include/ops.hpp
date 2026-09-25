#pragma once

#include "tensor.hpp"
#include <memory>
#include <vector>

namespace kittenml {

inline TensorPtr transpose(const TensorPtr &a) {
  auto out = std::make_shared<Tensor>(a->data.transpose());

  out->parents = {a};

  out->backward_fn = [a, out]() { a->grad += out->grad.transpose(); };

  return out;
}

inline TensorPtr relu(const TensorPtr &a) {
  auto out = std::make_shared<Tensor>(a->data.cwiseMax(0));

  out->parents = {a};

  out->backward_fn = [a, out]() {
    a->grad += (out->grad.array() > 0).select(out->grad, 0);
  };

  return out;
}
} // namespace kittenml