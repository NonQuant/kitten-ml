#include <kittenml/preprocessing.hpp>
#include <kittenml/data/split.hpp>
#include <kittenml/classical/linear_regression.hpp>
#include <kittenml/metrics.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Scaler uses training statistics on test data and constant columns", "[scaler]") {
  Eigen::MatrixXd train(3, 2), test(1, 2);
  train << 1, 7, 2, 7, 3, 7;
  test << 4, 9;
  kittenml::preprocessing::StandardScaler scaler;
  REQUIRE_THROWS_AS(scaler.transform(train), std::logic_error);
  const auto scaled = scaler.fit_transform(train);
  REQUIRE(scaled.colwise().mean().isZero(1e-12));
  REQUIRE(std::abs(scaled.col(0).squaredNorm() / 3 - 1) < 1e-12);
  REQUIRE(scaled.col(1).isZero());
  const auto transformed = scaler.transform(test);
  REQUIRE(std::abs(transformed(0, 0) - 2 / std::sqrt(2.0 / 3)) < 1e-12);
  REQUIRE(transformed(0, 1) == 2);
  REQUIRE(scaler.mean()(0) == 2);
  REQUIRE(scaler.scale()(1) == 1);
  REQUIRE_THROWS_AS(scaler.transform(Eigen::MatrixXd::Ones(2, 3)),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(scaler.fit(Eigen::MatrixXd(0, 2)), std::invalid_argument);
}

TEST_CASE("Split, scale, fit and evaluate a regression workflow", "[integration]") {
  Eigen::MatrixXd X(10, 1), y(10, 1);
  for (int i = 0; i < 10; ++i) { X(i, 0) = i; y(i, 0) = 3 * i + 2; }
  const auto split = kittenml::data::train_test_split(X, y, 0.3, false);
  kittenml::preprocessing::StandardScaler scaler;
  const auto train = scaler.fit_transform(split.X_train);
  const auto test = scaler.transform(split.X_test);
  kittenml::classical::LinearRegression model(0.1, 500);
  const auto prediction = model.fit(train, split.y_train).predict(test);
  REQUIRE(kittenml::metrics::mean_squared_error(split.y_test, prediction) < 1e-10);
  REQUIRE(kittenml::metrics::r2_score(split.y_test, prediction) > 0.99999);
}
