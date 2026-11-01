#include <kittenml/classical/decision_tree.hpp>
#include <iostream>

int main() {
  Eigen::MatrixXd X(6, 1);
  X << 0, 1, 2, 3, 4, 5;
  Eigen::VectorXi y(6);
  y << 0, 0, 1, 1, 0, 0;
  kittenml::classical::DecisionTreeClassifier model;
  const auto predictions = model.fit(X, y).predict(X);
  std::cout << "tree predictions: " << predictions.transpose() << '\n';
  return predictions == y ? 0 : 1;
}
