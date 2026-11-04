#pragma once

#include <Eigen/Dense>

#include <stdexcept>

namespace kittenml::metrics {

inline double mean_squared_error(const Eigen::MatrixXd &truth,
                                 const Eigen::MatrixXd &prediction) {
  if (truth.size() == 0)
    throw std::invalid_argument("metric inputs must not be empty");
  if (truth.rows() != prediction.rows() || truth.cols() != prediction.cols())
    throw std::invalid_argument("metric inputs must have the same shape");
  if (!truth.allFinite() || !prediction.allFinite())
    throw std::invalid_argument("metric inputs must contain only finite values");
  return (truth - prediction).array().square().mean();
}

// Uniform average of per-output R². For constant targets return 1 for
// perfect predictions and 0 otherwise. Negative scores are valid.
inline double r2_score(const Eigen::MatrixXd &truth,
                        const Eigen::MatrixXd &prediction) {
  if (truth.rows() < 2 || truth.cols() == 0 ||
      truth.rows() != prediction.rows() || truth.cols() != prediction.cols())
    throw std::invalid_argument("R2 requires matching shapes and at least two samples");
  if (!truth.allFinite() || !prediction.allFinite())
    throw std::invalid_argument("metric inputs must contain only finite values");
  double total = 0;
  for (Eigen::Index column = 0; column < truth.cols(); ++column) {
    const double residual = (truth.col(column) - prediction.col(column)).squaredNorm();
    const double variance =
        (truth.col(column).array() - truth.col(column).mean()).square().sum();
    total += variance == 0 ? (residual == 0 ? 1.0 : 0.0)
                           : 1.0 - residual / variance;
  }
  return total / truth.cols();
}

} // namespace kittenml::metrics
