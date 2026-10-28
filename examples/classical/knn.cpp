#include <kittenml/classical/knn.hpp>
#include <kittenml/metrics/classification.hpp>

#include <Eigen/Dense>
#include <iostream>

int main() {
  Eigen::MatrixXd X_train(6, 2);
  X_train << 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
             5.0, 5.0, 5.0, 6.0, 6.0, 5.0;
  Eigen::VectorXi y_train(6);
  y_train << 0, 0, 0, 1, 1, 1;

  Eigen::MatrixXd X_test(2, 2);
  X_test << 0.5, 0.5, 5.5, 5.5;
  Eigen::VectorXi expected(2);
  expected << 0, 1;

  kittenml::classical::KNNClassifier model(3);
  const Eigen::VectorXi predicted = model.fit(X_train, y_train).predict(X_test);
  std::cout << "predictions: " << predicted.transpose() << '\n';
  std::cout << "accuracy: "
            << kittenml::metrics::accuracy(expected, predicted) << '\n';
}
