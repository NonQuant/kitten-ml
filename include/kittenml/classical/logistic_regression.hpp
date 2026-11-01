#pragma once

#include <kittenml/detail/gradient_descent.hpp>
#include <kittenml/detail/validation.hpp>

namespace kittenml::classical {

// Binary classification with arbitrary integer class labels.
class LogisticRegression {
public:
  explicit LogisticRegression(double learning_rate = 0.1, int iterations = 1000)
      : learning_rate_(learning_rate), iterations_(iterations) {
    detail::validate_descent(learning_rate, iterations);
  }

  LogisticRegression &fit(const Eigen::MatrixXd &X, const Eigen::VectorXi &y) {
    detail::validate_supervised(X, y);
    const int low = y.minCoeff(), high = y.maxCoeff();
    if (low == high)
      throw std::invalid_argument("LogisticRegression requires two classes");
    Eigen::MatrixXd targets(y.size(), 1);
    for (Eigen::Index row = 0; row < y.size(); ++row) {
      if (y(row) != low && y(row) != high)
        throw std::invalid_argument("LogisticRegression supports only two classes");
      targets(row, 0) = y(row) == high ? 1.0 : 0.0;
    }
    weights_ = detail::train_linear_model(X, targets, learning_rate_, iterations_, true);
    low_ = low;
    high_ = high;
    return *this;
  }

  LogisticRegression &fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &y) {
    detail::validate_supervised(X, y);
    return fit(X, detail::class_labels(y));
  }

  // One probability per sample, for the larger of the two class labels.
  Eigen::VectorXd predict_proba(const Eigen::MatrixXd &X) const {
    if (!is_fitted()) throw std::logic_error("LogisticRegression must be fitted");
    detail::validate_feature_count(X, weights_.rows() - 1);
    auto logits = std::make_shared<Tensor>(
        (X * weights_.topRows(X.cols())).array() + weights_(X.cols(), 0));
    auto probabilities = sigmoid(logits);
    Eigen::VectorXd result = probabilities->data.col(0);
    detail::release_training_graph(probabilities);
    return result;
  }

  Eigen::VectorXi predict(const Eigen::MatrixXd &X) const {
    return predict_proba(X).unaryExpr([this](double p) {
      return p >= 0.5 ? high_ : low_;
    });
  }

  bool is_fitted() const noexcept { return weights_.size() > 0; }

private:
  double learning_rate_;
  int iterations_;
  int low_ = 0, high_ = 1;
  Eigen::MatrixXd weights_;
};

} // namespace kittenml::classical
