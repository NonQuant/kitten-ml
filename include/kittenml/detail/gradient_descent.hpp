#pragma once

#include <kittenml/nn/loss.hpp>
#include <kittenml/nn/sgd.hpp>
#include <kittenml/ops.hpp>

namespace kittenml::detail {

// Existing autograd callbacks own their outputs. Release a completed training
// graph explicitly so repeated estimator epochs do not retain those cycles.
inline void release_training_graph(const TensorPtr &node) {
  auto parents = std::move(node->parents);
  node->backward_fn = {};
  for (const auto &parent : parents)
    release_training_graph(parent);
}

inline void validate_descent(double learning_rate, int iterations) {
  if (!std::isfinite(learning_rate) || learning_rate <= 0 || iterations <= 0)
    throw std::invalid_argument("learning_rate and iterations must be positive");
}

inline Eigen::MatrixXd train_linear_model(const Eigen::MatrixXd &X,
                                          const Eigen::MatrixXd &y,
                                          double learning_rate, int iterations,
                                          bool logistic) {
  Eigen::MatrixXd design(X.rows(), X.cols() + 1);
  design.leftCols(X.cols()) = X;
  design.col(X.cols()).setOnes();
  auto weights = std::make_shared<Tensor>(
      Eigen::MatrixXd::Zero(design.cols(), y.cols()));
  nn::SGD optimizer({weights}, learning_rate);
  for (int epoch = 0; epoch < iterations; ++epoch) {
    auto input = std::make_shared<Tensor>(design);
    auto target = std::make_shared<Tensor>(y);
    auto logits = matmul(input, weights);
    TensorPtr loss;
    if (logistic) {
      auto probabilities = sigmoid(logits);
      Eigen::MatrixXd value(1, 1);
      value(0, 0) = logits->data.binaryExpr(y, [](double z, double t) {
        return std::max(z, 0.0) - z * t + std::log1p(std::exp(-std::abs(z)));
      }).mean();
      loss = std::make_shared<Tensor>(value);
      loss->parents = {logits};
      Eigen::MatrixXd derivative = (probabilities->data - y) / y.size();
      loss->backward_fn = [logits, loss, derivative]() {
        logits->grad += loss->grad(0, 0) * derivative;
      };
      // The stable BCE-with-logits derivative already includes sigmoid.
      probabilities->backward_fn = {};
      probabilities->parents.clear();
    } else {
      loss = nn::MSELoss()(logits, target);
    }
    loss->backward();
    optimizer.step();
    const bool finite = weights->data.allFinite();
    release_training_graph(loss);
    if (!finite)
      throw std::runtime_error("gradient descent diverged; reduce learning_rate");
  }
  return weights->data;
}

} // namespace kittenml::detail
