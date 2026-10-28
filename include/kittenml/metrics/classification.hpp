#pragma once

#include <Eigen/Dense>

#include <stdexcept>

namespace kittenml::metrics {

inline double accuracy(const Eigen::VectorXi &truth,
                       const Eigen::VectorXi &prediction) {
  if (truth.size() == 0)
    throw std::invalid_argument("metric inputs must not be empty");
  if (truth.size() != prediction.size())
    throw std::invalid_argument("metric inputs must have the same size");
  return (truth.array() == prediction.array()).cast<double>().mean();
}

} // namespace kittenml::metrics
