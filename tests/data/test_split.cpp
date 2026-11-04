#include <kittenml/data/split.hpp>
#include <catch2/catch_test_macros.hpp>
#include <set>

TEST_CASE("Split is seeded, disjoint and preserves integer label alignment", "[data][split]") {
  Eigen::MatrixXd X(10, 1);
  Eigen::VectorXi y(10);
  for (int i = 0; i < 10; ++i) { X(i, 0) = i; y(i) = i + 10; }
  const auto split = kittenml::data::train_test_split(X, y, 0.25, true, 42);
  const auto repeat = kittenml::data::train_test_split(X, y, 0.25, true, 42);
  REQUIRE(split.X_train.rows() == 7);
  REQUIRE(split.X_test.rows() == 3);
  REQUIRE(split.X_train == repeat.X_train);
  REQUIRE(split.X_test == repeat.X_test);
  std::set<int> seen;
  for (Eigen::Index i = 0; i < split.X_train.rows(); ++i) {
    REQUIRE(split.y_train(i) == split.X_train(i, 0) + 10);
    REQUIRE(seen.insert(static_cast<int>(split.X_train(i, 0))).second);
  }
  for (Eigen::Index i = 0; i < split.X_test.rows(); ++i) {
    REQUIRE(split.y_test(i) == split.X_test(i, 0) + 10);
    REQUIRE(seen.insert(static_cast<int>(split.X_test(i, 0))).second);
  }
  REQUIRE(seen.size() == 10);
}

TEST_CASE("Split supports matrix targets and ordered partitions", "[data][split]") {
  Eigen::MatrixXd X(5, 1), y(5, 2);
  X << 0, 1, 2, 3, 4;
  y.col(0) = X;
  y.col(1) = 2 * X;
  kittenml::Dataset dataset(X, y);
  const auto split = kittenml::data::train_test_split(dataset, 0.4, false);
  REQUIRE(split.X_train == X.topRows(3));
  REQUIRE(split.X_test == X.bottomRows(2));
  REQUIRE(split.y_test == y.bottomRows(2));
  REQUIRE_THROWS_AS(kittenml::data::train_test_split(dataset, 0), std::invalid_argument);
  REQUIRE_THROWS_AS(kittenml::data::train_test_split(dataset, 0.99), std::invalid_argument);
}
