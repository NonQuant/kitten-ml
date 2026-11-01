#include <kittenml/classical/logistic_regression.hpp>
#include <kittenml/metrics/classification.hpp>
#include <iostream>

int main() {
  Eigen::MatrixXd X(6, 1);
  X << -3, -2, -1, 1, 2, 3;
  Eigen::VectorXi y(6);
  y << 0, 0, 0, 1, 1, 1;
  kittenml::classical::LogisticRegression model;
  const double accuracy = kittenml::metrics::accuracy(
      y, model.fit(X, y).predict(X));
  std::cout << "logistic regression accuracy: " << accuracy << '\n';
  return accuracy == 1 ? 0 : 1;
}
