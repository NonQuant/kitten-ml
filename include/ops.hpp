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
} // namespace kittenml