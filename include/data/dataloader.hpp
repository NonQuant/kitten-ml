#pragma once

#include "dataset.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kittenml {

class DataLoader {
public:
  DataLoader(Dataset dataset, Eigen::Index batch_size, bool shuffle,
             std::uint32_t seed = 0)
      : dataset_(std::move(dataset)), batch_size_(batch_size),
        shuffle_(shuffle), generator_(seed) {
    if (batch_size_ <= 0)
      throw std::invalid_argument("batch_size must be positive");
    if (dataset_.X.rows() != dataset_.y.rows())
      throw std::invalid_argument(
          "X and y must have the same number of samples");

    indices_.resize(static_cast<std::size_t>(dataset_.size()));
    reset();
  }

  bool next(Batch &batch) {
    if (position_ == dataset_.size())
      return false;

    const Eigen::Index current_batch_size =
        std::min(batch_size_, dataset_.size() - position_);
    Eigen::MatrixXd X(current_batch_size, dataset_.X.cols());
    Eigen::MatrixXd y(current_batch_size, dataset_.y.cols());

    for (Eigen::Index row = 0; row < current_batch_size; ++row) {
      const Eigen::Index index =
          indices_[static_cast<std::size_t>(position_ + row)];
      X.row(row) = dataset_.X.row(index);
      y.row(row) = dataset_.y.row(index);
    }

    position_ += current_batch_size;
    batch.X = std::make_shared<Tensor>(std::move(X));
    batch.y = std::make_shared<Tensor>(std::move(y));
    return true;
  }

  void reset() {
    position_ = 0;
    std::iota(indices_.begin(), indices_.end(), Eigen::Index{0});
    if (shuffle_)
      std::shuffle(indices_.begin(), indices_.end(), generator_);
  }

private:
  Dataset dataset_;
  Eigen::Index batch_size_;
  bool shuffle_;
  std::mt19937 generator_;
  std::vector<Eigen::Index> indices_;
  Eigen::Index position_ = 0;
};

} // namespace kittenml
