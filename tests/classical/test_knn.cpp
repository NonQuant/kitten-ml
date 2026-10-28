#include <catch2/catch_test_macros.hpp>
#include <kittenml/classical/knn.hpp>

TEST_CASE("k-NN predicts by majority vote", "[classical][knn]") {
  Eigen::MatrixXd X(6, 2);
  X << 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
       5.0, 5.0, 5.0, 6.0, 6.0, 5.0;
  Eigen::VectorXi y(6);
  y << 0, 0, 0, 1, 1, 1;
  Eigen::MatrixXd queries(2, 2);
  queries << 0.25, 0.25, 5.25, 5.25;

  kittenml::classical::KNNClassifier model(3);
  Eigen::VectorXi expected(2);
  expected << 0, 1;
  REQUIRE(model.fit(X, y).predict(queries) == expected);
}

TEST_CASE("k-NN resolves vote ties using the smallest label",
          "[classical][knn]") {
  Eigen::MatrixXd X(2, 1);
  X << -1.0, 1.0;
  Eigen::VectorXi y(2);
  y << 7, 3;
  Eigen::MatrixXd query(1, 1);
  query << 0.0;

  REQUIRE(kittenml::classical::KNNClassifier(2)
              .fit(X, y)
              .predict(query)(0) == 3);
}

TEST_CASE("k-NN validates fitting and feature dimensions",
          "[classical][knn]") {
  kittenml::classical::KNNClassifier model(1);
  REQUIRE_THROWS_AS(model.predict(Eigen::MatrixXd::Ones(1, 1)),
                    std::logic_error);

  Eigen::MatrixXd X = Eigen::MatrixXd::Ones(2, 2);
  Eigen::VectorXi y(2);
  y << 0, 1;
  model.fit(X, y);
  REQUIRE_THROWS_AS(model.predict(Eigen::MatrixXd::Ones(1, 3)),
                    std::invalid_argument);
}
