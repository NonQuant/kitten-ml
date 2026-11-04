#pragma once

#include <kittenml/detail/validation.hpp>
#include <utility>

namespace kittenml::preprocessing {

class StandardScaler {
public:
  StandardScaler &fit(const Eigen::MatrixXd &X) {
    detail::validate_features(X);
    Eigen::RowVectorXd mean = X.colwise().mean();
    Eigen::RowVectorXd scale =
        (X.rowwise() - mean).array().square().colwise().mean().sqrt();
    if (!mean.allFinite() || !scale.allFinite())
      throw std::invalid_argument("scaler statistics overflowed");
    for (Eigen::Index column = 0; column < scale.size(); ++column)
      if (scale(column) == 0) scale(column) = 1;
    mean_ = std::move(mean);
    scale_ = std::move(scale);
    return *this;
  }

  Eigen::MatrixXd transform(const Eigen::MatrixXd &X) const {
    require_fitted();
    detail::validate_feature_count(X, mean_.size());
    return (X.rowwise() - mean_).array().rowwise() / scale_.array();
  }

  Eigen::MatrixXd fit_transform(const Eigen::MatrixXd &X) {
    return fit(X).transform(X);
  }

  const Eigen::RowVectorXd &mean() const {
    require_fitted();
    return mean_;
  }
  const Eigen::RowVectorXd &scale() const {
    require_fitted();
    return scale_;
  }
  bool is_fitted() const noexcept { return mean_.size() > 0; }

private:
  void require_fitted() const {
    if (!is_fitted()) throw std::logic_error("StandardScaler must be fitted");
  }
  Eigen::RowVectorXd mean_, scale_;
};

} // namespace kittenml::preprocessing
