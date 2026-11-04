#pragma once

#include <Eigen/Dense>

#include <stdexcept>
#include <map>
#include <set>

namespace kittenml::metrics {

inline double accuracy(const Eigen::VectorXi &truth,
                       const Eigen::VectorXi &prediction) {
  if (truth.size() == 0)
    throw std::invalid_argument("metric inputs must not be empty");
  if (truth.size() != prediction.size())
    throw std::invalid_argument("metric inputs must have the same size");
  return (truth.array() == prediction.array()).cast<double>().mean();
}

struct ConfusionMatrix {
  // counts(i, j): true class labels(i), predicted class labels(j).
  Eigen::MatrixXi counts;
  Eigen::VectorXi labels;
};

inline ConfusionMatrix confusion_matrix(const Eigen::VectorXi &truth,
                                        const Eigen::VectorXi &prediction) {
  if (truth.size() == 0 || truth.size() != prediction.size())
    throw std::invalid_argument("metric inputs must be nonempty and equal-sized");
  std::set<int> classes;
  for (Eigen::Index i = 0; i < truth.size(); ++i) {
    classes.insert(truth(i));
    classes.insert(prediction(i));
  }
  ConfusionMatrix result;
  result.labels.resize(classes.size());
  result.counts = Eigen::MatrixXi::Zero(classes.size(), classes.size());
  std::map<int, Eigen::Index> indices;
  Eigen::Index index = 0;
  for (int label : classes) {
    result.labels(index) = label;
    indices[label] = index++;
  }
  for (Eigen::Index i = 0; i < truth.size(); ++i)
    ++result.counts(indices.at(truth(i)), indices.at(prediction(i)));
  return result;
}

} // namespace kittenml::metrics
