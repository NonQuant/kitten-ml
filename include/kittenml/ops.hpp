#pragma once

#include <kittenml/tensor.hpp>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
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
    a->grad += (a->data.array() > 0).select(out->grad, 0);
  };

  return out;
}

inline TensorPtr sum(const TensorPtr &a) {
  Eigen::MatrixXd data(1, 1);
  data(0, 0) = a->data.sum();
  auto out = std::make_shared<Tensor>(data);

  out->parents = {a};

  out->backward_fn = [a, out]() {
    a->grad.array() += out->grad(0, 0);
  };

  return out;
}

inline TensorPtr sum(const TensorPtr &a, int axis) {
  Eigen::MatrixXd data;

  if (axis == 0)
    data = a->data.colwise().sum();
  else if (axis == 1)
    data = a->data.rowwise().sum();
  else
    throw std::invalid_argument("axis must be 0 or 1");

  auto out = std::make_shared<Tensor>(data);
  out->parents = {a};

  out->backward_fn = [a, out, axis]() {
    if (axis == 0)
      a->grad += out->grad.replicate(a->data.rows(), 1);
    else
      a->grad += out->grad.replicate(1, a->data.cols());
  };

  return out;
}

inline TensorPtr sigmoid(const TensorPtr &a) {
  auto data = a->data.unaryExpr([](double value) {
    if (value >= 0) {
      double z = std::exp(-value);
      return 1.0 / (1.0 + z);
    }

    double z = std::exp(value);
    return z / (1.0 + z);
  });
  auto out = std::make_shared<Tensor>(data);

  out->parents = {a};

  out->backward_fn = [a, out]() {
    a->grad += (out->grad.array() * out->data.array() *
                (1.0 - out->data.array()))
                   .matrix();
  };

  return out;
}

inline TensorPtr log(const TensorPtr &a) {
  constexpr double epsilon = std::numeric_limits<double>::epsilon();
  auto data = a->data.unaryExpr(
      [epsilon](double value) { return std::log(std::max(value, epsilon)); });
  auto out = std::make_shared<Tensor>(data);

  out->parents = {a};

  out->backward_fn = [a, out, epsilon]() {
    a->grad += (a->data.array() > epsilon)
                   .select(out->grad.array() / a->data.array(), 0.0)
                   .matrix();
  };

  return out;
}

inline TensorPtr exp(const TensorPtr &a) {
  auto out = std::make_shared<Tensor>(a->data.array().exp().matrix());

  out->parents = {a};

  out->backward_fn = [a, out]() {
    a->grad += out->grad.cwiseProduct(out->data);
  };

  return out;
}

inline TensorPtr softmax(const TensorPtr &a, int axis = 1) {
  Eigen::MatrixXd data(a->data.rows(), a->data.cols());

  if (axis == 0) {
    Eigen::RowVectorXd maxima = a->data.colwise().maxCoeff();
    Eigen::MatrixXd shifted = a->data.rowwise() - maxima;
    Eigen::RowVectorXd totals = shifted.array().exp().colwise().sum();
    data = shifted.array().exp().rowwise() / totals.array();
  } else if (axis == 1) {
    Eigen::VectorXd maxima = a->data.rowwise().maxCoeff();
    Eigen::MatrixXd shifted = a->data.colwise() - maxima;
    Eigen::VectorXd totals = shifted.array().exp().rowwise().sum();
    data = shifted.array().exp().colwise() / totals.array();
  } else {
    throw std::invalid_argument("axis must be 0 or 1");
  }

  auto out = std::make_shared<Tensor>(data);
  out->parents = {a};

  out->backward_fn = [a, out, axis]() {
    Eigen::MatrixXd product = out->grad.cwiseProduct(out->data);
    Eigen::MatrixXd centered = out->grad;

    if (axis == 0) {
      Eigen::RowVectorXd totals = product.colwise().sum();
      centered.rowwise() -= totals;
    } else {
      Eigen::VectorXd totals = product.rowwise().sum();
      centered.colwise() -= totals;
    }

    a->grad += out->data.cwiseProduct(centered);
  };

  return out;
}
} // namespace kittenml
