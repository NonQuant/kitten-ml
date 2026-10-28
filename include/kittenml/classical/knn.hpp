#pragma once

#include <kittenml/detail/distance.hpp>
#include <kittenml/detail/validation.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kittenml::classical {

class KNNClassifier {
public:
  explicit KNNClassifier(Eigen::Index n_neighbors = 5)
      : n_neighbors_(n_neighbors) {
    if (n_neighbors_ <= 0)
      throw std::invalid_argument("n_neighbors must be positive");
  }

  KNNClassifier &fit(const Eigen::MatrixXd &X, const Eigen::VectorXi &y) {
    detail::validate_supervised(X, y);
    if (n_neighbors_ > X.rows())
      throw std::invalid_argument(
          "n_neighbors must not exceed the number of samples");
    X_train_ = X;
    y_train_ = y;
    fitted_ = true;
    return *this;
  }

  KNNClassifier &fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &y) {
    detail::validate_supervised(X, y);
    return fit(X, detail::class_labels(y));
  }

  Eigen::VectorXi predict(const Eigen::MatrixXd &X) const {
    require_fitted();
    detail::validate_feature_count(X, X_train_.cols());
    const Eigen::MatrixXd distances =
        detail::squared_euclidean_distances(X, X_train_);
    Eigen::VectorXi predictions(X.rows());

    std::vector<Eigen::Index> indices(static_cast<std::size_t>(X_train_.rows()));
    std::iota(indices.begin(), indices.end(), Eigen::Index{0});
    for (Eigen::Index row = 0; row < X.rows(); ++row) {
      std::partial_sort(indices.begin(), indices.begin() + n_neighbors_,
                        indices.end(),
                        [&](Eigen::Index first, Eigen::Index second) {
                          if (distances(row, first) == distances(row, second))
                            return first < second;
                          return distances(row, first) < distances(row, second);
                        });

      std::map<int, int> votes;
      for (Eigen::Index neighbor = 0; neighbor < n_neighbors_; ++neighbor)
        ++votes[y_train_(indices[static_cast<std::size_t>(neighbor)])];

      int best_label = votes.begin()->first;
      int best_votes = votes.begin()->second;
      for (const auto &[label, count] : votes) {
        if (count > best_votes) {
          best_label = label;
          best_votes = count;
        }
      }
      predictions(row) = best_label;
    }
    return predictions;
  }

  Eigen::Index n_neighbors() const noexcept { return n_neighbors_; }
  bool is_fitted() const noexcept { return fitted_; }

private:
  void require_fitted() const {
    if (!fitted_)
      throw std::logic_error("KNNClassifier must be fitted before prediction");
  }

  Eigen::Index n_neighbors_;
  Eigen::MatrixXd X_train_;
  Eigen::VectorXi y_train_;
  bool fitted_ = false;
};

} // namespace kittenml::classical
