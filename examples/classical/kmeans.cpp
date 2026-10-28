#include <kittenml/classical/kmeans.hpp>

#include <Eigen/Dense>
#include <iostream>

int main() {
  Eigen::MatrixXd X(6, 2);
  X << -5.0, -5.0, -5.0, -4.0, -4.0, -5.0,
        5.0,  5.0,  5.0,  4.0,  4.0,  5.0;

  kittenml::classical::KMeans model(2, 100, 1e-8, 42);
  model.fit(X);
  std::cout << "centroids:\n" << model.centroids() << '\n';
  std::cout << "assignments: " << model.labels().transpose() << '\n';
  std::cout << "inertia: " << model.inertia() << '\n';
}
