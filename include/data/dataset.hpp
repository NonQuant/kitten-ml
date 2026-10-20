#pragma once

#include "tensor.hpp"
#include <Eigen/Dense>
#include <stdexcept>
#include <utility>

namespace kittenml {

struct Batch {
  TensorPtr X;
  TensorPtr y;
};

// X is (n_samples, n_features); y is (n_samples, n_targets).
struct Dataset {
  Eigen::MatrixXd X;
  Eigen::MatrixXd y;

  Dataset(Eigen::MatrixXd X, Eigen::MatrixXd y)
      : X(std::move(X)), y(std::move(y)) {
    if (this->X.rows() != this->y.rows())
      throw std::invalid_argument(
          "X and y must have the same number of samples");
  }

  Eigen::Index size() const noexcept { return X.rows(); }
};

} // namespace kittenml
