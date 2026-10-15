#pragma once

#include <Eigen/Dense>
#include <functional>
#include <memory>
#include <stdexcept>
#include <unordered_set>
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

    std::vector<Tensor *> topo_order;
    std::unordered_set<Tensor *> visited;

    build_topo(this, visited, topo_order); // post-order DFS

    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
      if ((*it)->backward_fn)
        (*it)->backward_fn();
    }
  }

private:
  static void build_topo(Tensor *node, std::unordered_set<Tensor *> &visited,
                         std::vector<Tensor *> &topo_order) {
    if (visited.count(node))
      return;
    visited.insert(node);

    for (auto &parent : node->parents) {
      build_topo(parent.get(), visited, topo_order);
    }

    topo_order.push_back(node);
  }
};

using TensorPtr = std::shared_ptr<Tensor>;

inline TensorPtr operator+(const TensorPtr &a, const TensorPtr &b) {
  bool broadcast_a = a->data.rows() == 1 &&
                     a->data.cols() == b->data.cols() && b->data.rows() > 1;
  bool broadcast_b = b->data.rows() == 1 &&
                     b->data.cols() == a->data.cols() && a->data.rows() > 1;
  Eigen::MatrixXd data;

  if (a->data.rows() == b->data.rows() &&
      a->data.cols() == b->data.cols())
    data = a->data + b->data;
  else if (broadcast_a)
    data = b->data.rowwise() + a->data.row(0);
  else if (broadcast_b)
    data = a->data.rowwise() + b->data.row(0);
  else
    throw std::invalid_argument("incompatible shapes for addition");

  auto out = std::make_shared<Tensor>(data);
  out->parents = {a, b};

  out->backward_fn = [a, b, out, broadcast_a, broadcast_b]() {
    if (broadcast_a)
      a->grad += out->grad.colwise().sum();
    else
      a->grad += out->grad;

    if (broadcast_b)
      b->grad += out->grad.colwise().sum();
    else
      b->grad += out->grad;
  };

  return out;
}

inline TensorPtr operator*(const TensorPtr &a, const TensorPtr &b) {
  auto out = std::make_shared<Tensor>(
      a->data.cwiseProduct(b->data)); // elementwise multiplication
  out->parents = {a, b};

  out->backward_fn = [a, b, out]() {
    a->grad += out->grad.cwiseProduct(b->data);
    b->grad += out->grad.cwiseProduct(a->data);
  };

  return out;
}

inline TensorPtr operator-(const TensorPtr &a, const TensorPtr &b) {
  auto out = std::make_shared<Tensor>(a->data - b->data);
  out->parents = {a, b};

  out->backward_fn = [a, b, out]() {
    a->grad += out->grad;
    b->grad += -out->grad;
  };

  return out;
}

inline TensorPtr operator/(const TensorPtr &a, const TensorPtr &b) {
  auto out = std::make_shared<Tensor>(a->data.cwiseQuotient(b->data));
  out->parents = {a, b};

  out->backward_fn = [a, b, out]() {
    a->grad += out->grad.cwiseQuotient(b->data); // out / b
    b->grad += -(out->grad.cwiseProduct(a->data.cwiseQuotient(
        b->data.cwiseProduct(b->data)))); // -out * a / b^2
  };

  return out;
}

inline TensorPtr matmul(const TensorPtr &a, const TensorPtr &b) {
  auto out = std::make_shared<Tensor>(a->data * b->data);
  out->parents = {a, b};

  out->backward_fn = [a, b, out]() {
    a->grad += out->grad * b->data.transpose();
    b->grad += a->data.transpose() * out->grad;
  };

  return out;
}

} // namespace kittenml
