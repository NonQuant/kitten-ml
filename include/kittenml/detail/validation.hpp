#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace kittenml::detail {

inline void validate_features(const Eigen::MatrixXd &X, const char *name = "X") {
  if (X.rows() == 0 || X.cols() == 0)
    throw std::invalid_argument(std::string(name) + " must not be empty");
  if (!X.allFinite())
    throw std::invalid_argument(std::string(name) +
                                " must contain only finite values");
}

inline void validate_feature_count(const Eigen::MatrixXd &X,
                                   Eigen::Index expected) {
  validate_features(X);
  if (X.cols() != expected)
    throw std::invalid_argument("X has an unexpected number of features");
}

inline void validate_supervised(const Eigen::MatrixXd &X,
                                const Eigen::VectorXi &y) {
  validate_features(X);
  if (y.size() == 0)
    throw std::invalid_argument("y must not be empty");
  if (X.rows() != y.size())
    throw std::invalid_argument("X and y must have the same number of samples");
}

inline void validate_supervised(const Eigen::MatrixXd &X,
                                const Eigen::MatrixXd &y) {
  validate_features(X);
  if (y.rows() == 0 || y.cols() == 0)
    throw std::invalid_argument("y must not be empty");
  if (!y.allFinite())
    throw std::invalid_argument("y must contain only finite values");
  if (X.rows() != y.rows())
    throw std::invalid_argument("X and y must have the same number of samples");
}

inline Eigen::VectorXi class_labels(const Eigen::MatrixXd &y) {
  if (y.cols() != 1)
    throw std::invalid_argument("classification targets must have one column");

  Eigen::VectorXi labels(y.rows());
  for (Eigen::Index row = 0; row < y.rows(); ++row) {
    const double value = y(row, 0);
    if (!std::isfinite(value) || std::floor(value) != value ||
        value < static_cast<double>(std::numeric_limits<int>::min()) ||
        value > static_cast<double>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("classification targets must be integer-valued");
    labels(row) = static_cast<int>(value);
  }
  return labels;
}

} // namespace kittenml::detail
