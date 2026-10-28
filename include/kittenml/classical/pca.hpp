#pragma once

#include <kittenml/detail/validation.hpp>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <stdexcept>

namespace kittenml::classical {

class PCA {
public:
  explicit PCA(Eigen::Index n_components) : n_components_(n_components) {
    if (n_components_ <= 0)
      throw std::invalid_argument("n_components must be positive");
  }

  PCA &fit(const Eigen::MatrixXd &X) {
    detail::validate_features(X);
    if (X.rows() < 2)
      throw std::invalid_argument("PCA requires at least two samples");
    if (n_components_ > X.cols())
      throw std::invalid_argument(
          "n_components must not exceed the number of features");

    mean_ = X.colwise().mean();
    const Eigen::MatrixXd centered = X.rowwise() - mean_;
    covariance_ = (centered.transpose() * centered) /
                  static_cast<double>(X.rows() - 1);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(covariance_);
    if (solver.info() != Eigen::Success)
      throw std::runtime_error("PCA eigendecomposition failed");

    components_.resize(n_components_, X.cols());
    explained_variance_.resize(n_components_);
    for (Eigen::Index component = 0; component < n_components_; ++component) {
      const Eigen::Index source = X.cols() - 1 - component;
      components_.row(component) = solver.eigenvectors().col(source).transpose();
      explained_variance_(component) = solver.eigenvalues()(source);
    }
    fitted_ = true;
    return *this;
  }

  Eigen::MatrixXd transform(const Eigen::MatrixXd &X) const {
    require_fitted();
    detail::validate_feature_count(X, components_.cols());
    return (X.rowwise() - mean_) * components_.transpose();
  }

  Eigen::MatrixXd fit_transform(const Eigen::MatrixXd &X) {
    fit(X);
    return transform(X);
  }

  const Eigen::MatrixXd &components() const {
    require_fitted();
    return components_;
  }

  const Eigen::VectorXd &explained_variance() const {
    require_fitted();
    return explained_variance_;
  }

  const Eigen::MatrixXd &covariance() const {
    require_fitted();
    return covariance_;
  }

  const Eigen::RowVectorXd &mean() const {
    require_fitted();
    return mean_;
  }

  bool is_fitted() const noexcept { return fitted_; }

private:
  void require_fitted() const {
    if (!fitted_)
      throw std::logic_error("PCA must be fitted before this operation");
  }

  Eigen::Index n_components_;
  Eigen::RowVectorXd mean_;
  Eigen::MatrixXd covariance_;
  Eigen::MatrixXd components_;
  Eigen::VectorXd explained_variance_;
  bool fitted_ = false;
};

} // namespace kittenml::classical
