#pragma once

#include <kittenml/detail/gradient_descent.hpp>
#include <kittenml/detail/validation.hpp>

namespace kittenml::classical {

class LinearRegression {
public:
  explicit LinearRegression(double learning_rate = 0.01, int iterations = 1000)
      : learning_rate_(learning_rate), iterations_(iterations) {
    detail::validate_descent(learning_rate, iterations);
  }

  LinearRegression &fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &y) {
    detail::validate_supervised(X, y);
    weights_ = detail::train_linear_model(X, y, learning_rate_, iterations_, false);
    return *this;
  }

  Eigen::MatrixXd predict(const Eigen::MatrixXd &X) const {
    if (!is_fitted()) throw std::logic_error("LinearRegression must be fitted");
    detail::validate_feature_count(X, weights_.rows() - 1);
    return (X * weights_.topRows(X.cols())).rowwise() +
           weights_.bottomRows(1).row(0);
  }

  bool is_fitted() const noexcept { return weights_.size() > 0; }

private:
  double learning_rate_;
  int iterations_;
  Eigen::MatrixXd weights_;
};

} // namespace kittenml::classical
