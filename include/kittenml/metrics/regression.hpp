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

} // namespace kittenml::metrics
