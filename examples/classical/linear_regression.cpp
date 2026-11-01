#include <kittenml/classical/linear_regression.hpp>
#include <kittenml/metrics/regression.hpp>
#include <iostream>

int main() {
  Eigen::MatrixXd X(5, 1), y(5, 1);
  X << -2, -1, 0, 1, 2;
  y = (2 * X.array() + 3).matrix();
  kittenml::classical::LinearRegression model(0.1, 500);
  const double mse = kittenml::metrics::mean_squared_error(
      y, model.fit(X, y).predict(X));
  std::cout << "linear regression MSE: " << mse << '\n';
  return mse < 1e-8 ? 0 : 1;
}
