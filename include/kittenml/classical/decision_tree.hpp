#pragma once

#include <kittenml/detail/validation.hpp>
#include <algorithm>
#include <map>
#include <memory>
#include <numeric>
#include <vector>

namespace kittenml::classical {

// Classification tree using weighted Gini impurity.
class DecisionTreeClassifier {
  struct Node {
    int label = 0;
    Eigen::Index feature = -1;
    double threshold = 0;
    std::unique_ptr<Node> left, right;
  };

public:
  explicit DecisionTreeClassifier(int max_depth = 20, int min_samples_split = 2)
      : max_depth_(max_depth), min_samples_split_(min_samples_split) {
    if (max_depth < 0 || min_samples_split < 2)
      throw std::invalid_argument("invalid tree stopping parameters");
  }

  DecisionTreeClassifier &fit(const Eigen::MatrixXd &X,
                              const Eigen::VectorXi &y) {
    detail::validate_supervised(X, y);
    std::vector<Eigen::Index> rows(static_cast<std::size_t>(X.rows()));
    std::iota(rows.begin(), rows.end(), Eigen::Index{0});
    auto root = build(X, y, rows, 0);
    root_ = std::move(root);
    n_features_ = X.cols();
    return *this;
  }

  DecisionTreeClassifier &fit(const Eigen::MatrixXd &X,
                              const Eigen::MatrixXd &y) {
    detail::validate_supervised(X, y);
    return fit(X, detail::class_labels(y));
  }

  Eigen::VectorXi predict(const Eigen::MatrixXd &X) const {
    if (!root_)
      throw std::logic_error("DecisionTreeClassifier must be fitted");
    detail::validate_feature_count(X, n_features_);
    Eigen::VectorXi result(X.rows());
    for (Eigen::Index row = 0; row < X.rows(); ++row) {
      const Node *node = root_.get();
      while (node->feature >= 0)
        node = X(row, node->feature) <= node->threshold
                   ? node->left.get() : node->right.get();
      result(row) = node->label;
    }
    return result;
  }

  bool is_fitted() const noexcept { return static_cast<bool>(root_); }

private:
  static double gini(const std::map<int, int> &counts, std::size_t size) {
    if (size == 0) return 0;
    double value = 1;
    for (const auto &[label, count] : counts) {
      const double p = static_cast<double>(count) / size;
      value -= p * p;
    }
    return value;
  }

  std::unique_ptr<Node> build(const Eigen::MatrixXd &X,
                              const Eigen::VectorXi &y,
                              const std::vector<Eigen::Index> &rows,
                              int depth) const {
    auto node = std::make_unique<Node>();
    std::map<int, int> counts;
    for (auto row : rows) ++counts[y(row)];
    int majority = -1;
    for (const auto &[label, count] : counts)
      if (count > majority) { majority = count; node->label = label; }
    if (counts.size() == 1 || depth >= max_depth_ ||
        rows.size() < static_cast<std::size_t>(min_samples_split_))
      return node;

    double best = gini(counts, rows.size());
    for (Eigen::Index feature = 0; feature < X.cols(); ++feature) {
      auto sorted = rows;
      std::sort(sorted.begin(), sorted.end(), [&](auto a, auto b) {
        return X(a, feature) < X(b, feature);
      });
      std::map<int, int> left, right = counts;
      for (std::size_t i = 1; i < sorted.size(); ++i) {
        ++left[y(sorted[i - 1])];
        --right[y(sorted[i - 1])];
        const double lower = X(sorted[i - 1], feature);
        const double upper = X(sorted[i], feature);
        if (lower == upper) continue;
        const double score =
            (i * gini(left, i) + (rows.size() - i) *
             gini(right, rows.size() - i)) / rows.size();
        if (score < best) {
          best = score;
          node->feature = feature;
          node->threshold = lower / 2 + upper / 2;
          if (node->threshold >= upper) node->threshold = lower;
        }
      }
    }
    if (node->feature < 0) return node;
    std::vector<Eigen::Index> left, right;
    for (auto row : rows)
      (X(row, node->feature) <= node->threshold ? left : right).push_back(row);
    node->left = build(X, y, left, depth + 1);
    node->right = build(X, y, right, depth + 1);
    return node;
  }

  int max_depth_, min_samples_split_;
  Eigen::Index n_features_ = 0;
  std::unique_ptr<Node> root_;
};

} // namespace kittenml::classical
