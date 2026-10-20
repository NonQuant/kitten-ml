#pragma once

#include "dataset.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kittenml {

inline Dataset make_xor() {
  Eigen::MatrixXd X(4, 2);
  X << 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0;

  Eigen::MatrixXd y(4, 1);
  y << 0.0, 1.0, 1.0, 0.0;

  return {std::move(X), std::move(y)};
}

inline Dataset make_blobs(Eigen::Index n_samples, Eigen::Index n_centers,
                          std::uint32_t seed) {
  if (n_samples < 0)
    throw std::invalid_argument("n_samples must not be negative");
  if (n_centers <= 0)
    throw std::invalid_argument("n_centers must be positive");

  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> center_distribution(-10.0, 10.0);
  std::normal_distribution<double> noise_distribution(0.0, 1.0);

  Eigen::MatrixXd centers(n_centers, 2);
  for (Eigen::Index row = 0; row < n_centers; ++row) {
    for (Eigen::Index column = 0; column < centers.cols(); ++column)
      centers(row, column) = center_distribution(generator);
  }

  std::vector<Eigen::Index> labels(static_cast<std::size_t>(n_samples));
  for (Eigen::Index sample = 0; sample < n_samples; ++sample)
    labels[static_cast<std::size_t>(sample)] = sample % n_centers;
  std::shuffle(labels.begin(), labels.end(), generator);

  Eigen::MatrixXd X(n_samples, 2);
  Eigen::MatrixXd y(n_samples, 1);
  for (Eigen::Index sample = 0; sample < n_samples; ++sample) {
    const Eigen::Index label = labels[static_cast<std::size_t>(sample)];
    X.row(sample) = centers.row(label);
    for (Eigen::Index feature = 0; feature < X.cols(); ++feature)
      X(sample, feature) += noise_distribution(generator);
    y(sample, 0) = static_cast<double>(label);
  }

  return {std::move(X), std::move(y)};
}

} // namespace kittenml
