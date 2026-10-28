#include <kittenml/classical/pca.hpp>

#include <Eigen/Dense>
#include <iostream>

int main() {
  Eigen::MatrixXd X(5, 2);
  X << -2.0, -2.0, -1.0, -1.0, 0.0, 0.0,
        1.0,  1.0,  2.0,  2.0;

  kittenml::classical::PCA model(1);
  const Eigen::MatrixXd projected = model.fit_transform(X);
  std::cout << "principal direction: " << model.components() << '\n';
  std::cout << "explained variance: "
            << model.explained_variance().transpose() << '\n';
  std::cout << "projected data:\n" << projected << '\n';
}
