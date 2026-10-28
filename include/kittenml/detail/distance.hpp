#pragma once

#include <Eigen/Dense>

namespace kittenml::detail {

// Returns D(i, j) = ||left.row(i) - right.row(j)||^2.
inline Eigen::MatrixXd squared_euclidean_distances(
    const Eigen::MatrixXd &left, const Eigen::MatrixXd &right) {
  Eigen::VectorXd left_norms = left.rowwise().squaredNorm();
  Eigen::VectorXd right_norms = right.rowwise().squaredNorm();
  Eigen::MatrixXd distances =
      left_norms.replicate(1, right.rows()) +
      right_norms.transpose().replicate(left.rows(), 1) -
      2.0 * left * right.transpose();
  // Roundoff can make mathematically zero distances slightly negative.
  return distances.cwiseMax(0.0);
}

} // namespace kittenml::detail
