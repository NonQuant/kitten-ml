#pragma once

#include <kittenml/tensor.hpp>

#include <cmath>
#include <stdexcept>

namespace kittenml::nn {

class MSELoss {
public:
  TensorPtr forward(const TensorPtr &prediction,
                    const TensorPtr &target) const {
    if (!prediction || !target)
      throw std::invalid_argument("MSELoss inputs must not be null");
    if (prediction->data.rows() != target->data.rows() ||
        prediction->data.cols() != target->data.cols())
      throw std::invalid_argument(
          "MSELoss prediction and target must have the same shape");
    if (prediction->data.size() == 0)
      throw std::invalid_argument("MSELoss inputs must not be empty");

    const Eigen::MatrixXd difference = prediction->data - target->data;
    Eigen::MatrixXd value(1, 1);
    value(0, 0) = difference.array().square().mean();

    auto out = std::make_shared<Tensor>(value);
    out->parents = {prediction, target};
    out->backward_fn = [prediction, target, out, difference]() {
      const double scale = 2.0 * out->grad(0, 0) /
                           static_cast<double>(difference.size());
      prediction->grad += scale * difference;
      target->grad -= scale * difference;
    };
    return out;
  }

  TensorPtr operator()(const TensorPtr &prediction,
                       const TensorPtr &target) const {
    return forward(prediction, target);
  }
};

class CrossEntropyLoss {
public:
  TensorPtr forward(const TensorPtr &logits, const TensorPtr &target) const {
    if (!logits || !target)
      throw std::invalid_argument("CrossEntropyLoss inputs must not be null");
    if (logits->data.rows() == 0 || logits->data.cols() == 0)
      throw std::invalid_argument(
          "CrossEntropyLoss inputs must not be empty");

    const bool class_indices = target->data.rows() == logits->data.rows() &&
                               target->data.cols() == 1;
    const bool distributions =
        target->data.rows() == logits->data.rows() &&
        target->data.cols() == logits->data.cols();
    if (!class_indices && !distributions)
      throw std::invalid_argument(
          "CrossEntropyLoss target must contain one class index per row or "
          "have the same shape as logits");

    Eigen::MatrixXd probabilities(logits->data.rows(), logits->data.cols());
    Eigen::MatrixXd log_probabilities(logits->data.rows(),
                                      logits->data.cols());

    for (Eigen::Index row = 0; row < logits->data.rows(); ++row) {
      const double maximum = logits->data.row(row).maxCoeff();
      const Eigen::ArrayXd exponentials =
          (logits->data.row(row).array() - maximum).exp();
      const double log_total = std::log(exponentials.sum());
      log_probabilities.row(row) =
          logits->data.row(row).array() - maximum - log_total;
      probabilities.row(row) = log_probabilities.row(row).array().exp();
    }

    Eigen::MatrixXd effective_target =
        Eigen::MatrixXd::Zero(logits->data.rows(), logits->data.cols());
    if (class_indices) {
      for (Eigen::Index row = 0; row < target->data.rows(); ++row) {
        const double value = target->data(row, 0);
        if (!std::isfinite(value) || std::floor(value) != value || value < 0.0 ||
            value >= static_cast<double>(logits->data.cols()))
          throw std::invalid_argument(
              "CrossEntropyLoss class index is out of range");
        effective_target(row, static_cast<Eigen::Index>(value)) = 1.0;
      }
    } else {
      if (!target->data.allFinite() || (target->data.array() < 0.0).any())
        throw std::invalid_argument(
            "CrossEntropyLoss targets must be finite and non-negative");
      effective_target = target->data;
    }

    const double batch_size = static_cast<double>(logits->data.rows());
    Eigen::MatrixXd value(1, 1);
    value(0, 0) =
        -(effective_target.array() * log_probabilities.array()).sum() /
        batch_size;

    auto out = std::make_shared<Tensor>(value);
    out->parents = {logits, target};
    out->backward_fn = [logits, target, out, probabilities,
                        log_probabilities, effective_target, class_indices,
                        batch_size]() {
      const Eigen::VectorXd target_totals =
          effective_target.rowwise().sum();
      Eigen::MatrixXd logits_gradient = probabilities.array().colwise() *
                                        target_totals.array();
      logits_gradient -= effective_target;
      logits->grad += (out->grad(0, 0) / batch_size) * logits_gradient;

      // Integer class labels are discrete and therefore have no derivative.
      if (!class_indices)
        target->grad -=
            (out->grad(0, 0) / batch_size) * log_probabilities;
    };
    return out;
  }

  TensorPtr operator()(const TensorPtr &logits, const TensorPtr &target) const {
    return forward(logits, target);
  }
};

} // namespace kittenml::nn
