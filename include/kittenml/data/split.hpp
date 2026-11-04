#pragma once

#include <kittenml/data/dataset.hpp>
#include <kittenml/detail/validation.hpp>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace kittenml::data {

template <typename Targets> struct TrainTestSplit {
  Eigen::MatrixXd X_train, X_test;
  Targets y_train, y_test;
};

namespace split_detail {
template <typename Targets>
inline TrainTestSplit<Targets> split(const Eigen::MatrixXd &X, const Targets &y,
                                     double test_size, bool shuffle,
                                     std::uint32_t seed) {
  detail::validate_supervised(X, y);
  if (!std::isfinite(test_size) || test_size <= 0 || test_size >= 1)
    throw std::invalid_argument("test_size must be between zero and one");
  const Eigen::Index n_test =
      static_cast<Eigen::Index>(std::ceil(test_size * X.rows()));
  const Eigen::Index n_train = X.rows() - n_test;
  if (n_train < 1 || n_test < 1)
    throw std::invalid_argument("split must contain training and test samples");
  std::vector<Eigen::Index> indices(static_cast<std::size_t>(X.rows()));
  std::iota(indices.begin(), indices.end(), Eigen::Index{0});
  if (shuffle) {
    std::mt19937 generator(seed);
    std::shuffle(indices.begin(), indices.end(), generator);
  }
  TrainTestSplit<Targets> result;
  result.X_train.resize(n_train, X.cols());
  result.X_test.resize(n_test, X.cols());
  result.y_train.resize(n_train, y.cols());
  result.y_test.resize(n_test, y.cols());
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const auto source = indices[static_cast<std::size_t>(i)];
    if (i < n_train) {
      result.X_train.row(i) = X.row(source);
      result.y_train.row(i) = y.row(source);
    } else {
      result.X_test.row(i - n_train) = X.row(source);
      result.y_test.row(i - n_train) = y.row(source);
    }
  }
  return result;
}
} // namespace split_detail

inline TrainTestSplit<Eigen::MatrixXd>
train_test_split(const Eigen::MatrixXd &X, const Eigen::MatrixXd &y,
                 double test_size = 0.2, bool shuffle = true,
                 std::uint32_t seed = 0) {
  return split_detail::split(X, y, test_size, shuffle, seed);
}

inline TrainTestSplit<Eigen::VectorXi>
train_test_split(const Eigen::MatrixXd &X, const Eigen::VectorXi &y,
                 double test_size = 0.2, bool shuffle = true,
                 std::uint32_t seed = 0) {
  return split_detail::split(X, y, test_size, shuffle, seed);
}

inline TrainTestSplit<Eigen::MatrixXd>
train_test_split(const Dataset &dataset, double test_size = 0.2,
                 bool shuffle = true, std::uint32_t seed = 0) {
  return train_test_split(dataset.X, dataset.y, test_size, shuffle, seed);
}

} // namespace kittenml::data
