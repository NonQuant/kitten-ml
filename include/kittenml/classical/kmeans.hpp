#pragma once

#include <kittenml/detail/distance.hpp>
#include <kittenml/detail/validation.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kittenml::classical {

class KMeans {
public:
  explicit KMeans(Eigen::Index n_clusters, Eigen::Index max_iterations = 300,
                  double tolerance = 1e-4, std::uint32_t seed = 0)
      : n_clusters_(n_clusters), max_iterations_(max_iterations),
        tolerance_(tolerance), seed_(seed) {
    if (n_clusters_ <= 0)
      throw std::invalid_argument("n_clusters must be positive");
    if (max_iterations_ <= 0)
      throw std::invalid_argument("max_iterations must be positive");
    if (!std::isfinite(tolerance_) || tolerance_ < 0.0)
      throw std::invalid_argument("tolerance must be finite and non-negative");
  }

  KMeans &fit(const Eigen::MatrixXd &X) {
    detail::validate_features(X);
    if (n_clusters_ > X.rows())
      throw std::invalid_argument(
          "n_clusters must not exceed the number of samples");

    initialize_centroids(X);
    n_iterations_ = 0;
    for (Eigen::Index iteration = 0; iteration < max_iterations_; ++iteration) {
      labels_ = assign(X, centroids_);
      Eigen::MatrixXd updated = update_centroids(X, labels_, centroids_);
      const double maximum_shift =
          (updated - centroids_).rowwise().norm().maxCoeff();
      centroids_ = std::move(updated);
      n_iterations_ = iteration + 1;
      if (maximum_shift <= tolerance_)
        break;
    }

    // Labels and inertia must describe the final, updated centroids.
    labels_ = assign(X, centroids_);
    const Eigen::MatrixXd distances =
        detail::squared_euclidean_distances(X, centroids_);
    inertia_ = 0.0;
    for (Eigen::Index row = 0; row < X.rows(); ++row)
      inertia_ += distances(row, labels_(row));
    fitted_ = true;
    return *this;
  }

  Eigen::VectorXi predict(const Eigen::MatrixXd &X) const {
    require_fitted();
    detail::validate_feature_count(X, centroids_.cols());
    return assign(X, centroids_);
  }

  Eigen::VectorXi fit_predict(const Eigen::MatrixXd &X) {
    fit(X);
    return labels_;
  }

  const Eigen::MatrixXd &centroids() const {
    require_fitted();
    return centroids_;
  }

  const Eigen::VectorXi &labels() const {
    require_fitted();
    return labels_;
  }

  double inertia() const {
    require_fitted();
    return inertia_;
  }

  Eigen::Index n_iterations() const noexcept { return n_iterations_; }
  bool is_fitted() const noexcept { return fitted_; }

private:
  void initialize_centroids(const Eigen::MatrixXd &X) {
    std::vector<Eigen::Index> indices(static_cast<std::size_t>(X.rows()));
    std::iota(indices.begin(), indices.end(), Eigen::Index{0});
    std::mt19937 generator(seed_);
    std::shuffle(indices.begin(), indices.end(), generator);

    centroids_.resize(n_clusters_, X.cols());
    for (Eigen::Index cluster = 0; cluster < n_clusters_; ++cluster)
      centroids_.row(cluster) = X.row(indices[static_cast<std::size_t>(cluster)]);
  }

  static Eigen::VectorXi assign(const Eigen::MatrixXd &X,
                                const Eigen::MatrixXd &centroids) {
    const Eigen::MatrixXd distances =
        detail::squared_euclidean_distances(X, centroids);
    Eigen::VectorXi labels(X.rows());
    for (Eigen::Index row = 0; row < X.rows(); ++row) {
      Eigen::Index cluster = 0;
      distances.row(row).minCoeff(&cluster);
      labels(row) = static_cast<int>(cluster);
    }
    return labels;
  }

  Eigen::MatrixXd update_centroids(const Eigen::MatrixXd &X,
                                   const Eigen::VectorXi &labels,
                                   const Eigen::MatrixXd &previous) const {
    Eigen::MatrixXd updated = Eigen::MatrixXd::Zero(n_clusters_, X.cols());
    Eigen::VectorXi counts = Eigen::VectorXi::Zero(n_clusters_);
    for (Eigen::Index row = 0; row < X.rows(); ++row) {
      updated.row(labels(row)) += X.row(row);
      ++counts(labels(row));
    }
    for (Eigen::Index cluster = 0; cluster < n_clusters_; ++cluster) {
      if (counts(cluster) == 0)
        updated.row(cluster) = previous.row(cluster);
      else
        updated.row(cluster) /= static_cast<double>(counts(cluster));
    }
    return updated;
  }

  void require_fitted() const {
    if (!fitted_)
      throw std::logic_error("KMeans must be fitted before this operation");
  }

  Eigen::Index n_clusters_;
  Eigen::Index max_iterations_;
  double tolerance_;
  std::uint32_t seed_;
  Eigen::MatrixXd centroids_;
  Eigen::VectorXi labels_;
  Eigen::Index n_iterations_ = 0;
  double inertia_ = 0.0;
  bool fitted_ = false;
};

} // namespace kittenml::classical
