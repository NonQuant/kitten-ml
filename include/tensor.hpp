#pragma once

#include <Eigen/Dense>
#include <functional>
#include <memory>
#include <vector>

namespace kittenml {

// A Tensor wraps an Eigen matrix and keeps track of how it was
// produced, so that calling backward() can propagate gradients
// back through the graph it participated in.
class Tensor : public std::enable_shared_from_this<Tensor> {
public:
  Eigen::MatrixXd data;
  Eigen::MatrixXd grad;

  explicit Tensor(Eigen::MatrixXd data_)
      : data(std::move(data_)),
        grad(Eigen::MatrixXd::Zero(data.rows(), data.cols())) {}

  // TODO: add operator+, operator*, matmul(), etc. here.
  // TODO: add backward() here.
};

using TensorPtr = std::shared_ptr<Tensor>;

} // namespace kittenml
