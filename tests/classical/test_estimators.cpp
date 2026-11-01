#include <kittenml/classical.hpp>
#include <catch2/catch_test_macros.hpp>
#include <limits>

TEST_CASE("Decision tree learns recursive splits and arbitrary labels", "[tree]") {
  Eigen::MatrixXd X(6, 1);
  X << 0, 1, 2, 3, 4, 5;
  Eigen::VectorXi y(6);
  y << -4, -4, 7, 7, -4, -4;
  kittenml::classical::DecisionTreeClassifier tree;
  REQUIRE_THROWS_AS(tree.predict(X), std::logic_error);
  REQUIRE(tree.fit(X, y).predict(X) == y);
  kittenml::classical::DecisionTreeClassifier stump(0);
  REQUIRE((stump.fit(X, y).predict(X).array() == -4).all());
  X.setOnes();
  REQUIRE((tree.fit(X, y).predict(X).array() == -4).all());
  REQUIRE_THROWS_AS(tree.predict(Eigen::MatrixXd::Ones(1, 2)),
                    std::invalid_argument);
}

TEST_CASE("Linear regression learns slopes and intercepts with autograd", "[regression]") {
  Eigen::MatrixXd X(5, 1);
  X << -2, -1, 0, 1, 2;
  Eigen::MatrixXd y(5, 2);
  y.col(0) = (2 * X.array() + 3).matrix();
  y.col(1) = (-X.array() - 2).matrix();
  kittenml::classical::LinearRegression model(0.1, 500);
  REQUIRE_THROWS_AS(model.predict(X), std::logic_error);
  REQUIRE(model.fit(X, y).predict(X).isApprox(y, 1e-7));
  Eigen::MatrixXd query(2, 1), expected(2, 2);
  query << -3, 3;
  expected << -3, 1, 9, -5;
  REQUIRE(model.predict(query).isApprox(expected, 1e-7));
  REQUIRE(model.fit(X, y).predict(X).isApprox(y, 1e-7));
}

TEST_CASE("Logistic regression learns binary labels and sigmoid probabilities", "[logistic]") {
  Eigen::MatrixXd X(6, 1);
  X << -3, -2, -1, 1, 2, 3;
  Eigen::VectorXi y(6);
  y << -5, -5, -5, 9, 9, 9;
  kittenml::classical::LogisticRegression model;
  REQUIRE_THROWS_AS(model.predict(X), std::logic_error);
  REQUIRE(model.fit(X, y).predict(X) == y);
  const auto p = model.predict_proba(X);
  REQUIRE(p.allFinite());
  REQUIRE((p.array() > 0).all());
  REQUIRE((p.array() < 1).all());
  REQUIRE(p(0) < p(5));
  Eigen::MatrixXd extreme(2, 1);
  extreme << -10000, 10000;
  REQUIRE(model.predict_proba(extreme).allFinite());
  y(0) = 2;
  REQUIRE_THROWS_AS(model.fit(X, y), std::invalid_argument);
}

TEST_CASE("Gradient estimators reject invalid inputs", "[regression][logistic]") {
  REQUIRE_THROWS_AS(kittenml::classical::LinearRegression(0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(kittenml::classical::LogisticRegression(0.1, 0),
                    std::invalid_argument);
  Eigen::MatrixXd X = Eigen::MatrixXd::Ones(3, 1);
  Eigen::MatrixXd y = Eigen::MatrixXd::Zero(3, 1);
  kittenml::classical::LogisticRegression logistic;
  REQUIRE_THROWS_AS(logistic.fit(X, y), std::invalid_argument);
  y(0, 0) = 0.5;
  REQUIRE_THROWS_AS(logistic.fit(X, y), std::invalid_argument);
  X(0, 0) = std::numeric_limits<double>::quiet_NaN();
  kittenml::classical::LinearRegression linear;
  REQUIRE_THROWS_AS(linear.fit(X, y), std::invalid_argument);
}
