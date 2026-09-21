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

  // tensors that produced this tensor
  std::vector<std::shared_ptr<Tensor>> parents;

  // function for gradient math of specific operation that created this tensor
  std::function<void()> backward_fn;

  explicit Tensor(Eigen::MatrixXd data_)
      : data(std::move(data_)),
        grad(Eigen::MatrixXd::Zero(data.rows(), data.cols())) {}

  void backward() {
    grad = Eigen::MatrixXd::Ones(data.rows(), data.cols());
    backward_impl();
  }

private:
  // walk the graph backwards
  void backward_impl() {
    if (backward_fn) {
      backward_fn();
    }
    for (auto &parent : parents) {
      parent->backward_impl();
    }
  }
};

using TensorPtr = std::shared_ptr<Tensor>;

inline TensorPtr operator+(const TensorPtr &a, const TensorPtr &b) {
  auto out = std::make_shared<Tensor>(a->data + b->data);
  out->parents = {a, b};

  out->backward_fn = [a, b, out]() {
    a->grad += out->grad;
    b->grad += out->grad;
  };

  return out;
}

} // namespace kittenml
