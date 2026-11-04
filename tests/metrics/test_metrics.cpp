#include <kittenml/metrics.hpp>
#include <catch2/catch_test_macros.hpp>
#include <limits>

TEST_CASE("Classification metrics retain arbitrary and predicted-only labels", "[metrics]") {
  Eigen::VectorXi truth(4), prediction(4);
  truth << -5, -5, 9, 9;
  prediction << -5, 9, 9, 20;
  REQUIRE(kittenml::metrics::accuracy(truth, prediction) == 0.5);
  const auto confusion = kittenml::metrics::confusion_matrix(truth, prediction);
  Eigen::Vector3i labels;
  labels << -5, 9, 20;
  Eigen::Matrix3i counts;
  counts << 1, 1, 0, 0, 1, 1, 0, 0, 0;
  REQUIRE(confusion.labels == labels);
  REQUIRE(confusion.counts == counts);
  REQUIRE_THROWS_AS(kittenml::metrics::accuracy(Eigen::VectorXi(), prediction),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(kittenml::metrics::confusion_matrix(truth, Eigen::VectorXi()),
                    std::invalid_argument);
}

TEST_CASE("Regression metrics match hand-computed and multi-output scores", "[metrics]") {
  Eigen::MatrixXd truth(3, 1), prediction(3, 1);
  truth << 1, 2, 3;
  prediction << 1, 2, 2;
  REQUIRE(std::abs(kittenml::metrics::mean_squared_error(truth, prediction) - 1.0/3) < 1e-12);
  REQUIRE(kittenml::metrics::r2_score(truth, prediction) == 0.5);
  REQUIRE(kittenml::metrics::r2_score(truth, truth) == 1);
  Eigen::MatrixXd multi_truth(3, 2), multi_prediction(3, 2);
  multi_truth.col(0) = truth;
  multi_truth.col(1).setConstant(7);
  multi_prediction.col(0) = prediction;
  multi_prediction.col(1).setConstant(6);
  REQUIRE(kittenml::metrics::r2_score(multi_truth, multi_prediction) == 0.25);
  REQUIRE(kittenml::metrics::r2_score(multi_truth, multi_truth) == 1);
  prediction.setConstant(20);
  REQUIRE(kittenml::metrics::r2_score(truth, prediction) < 0);
  REQUIRE_THROWS_AS(kittenml::metrics::r2_score(Eigen::MatrixXd::Ones(1, 1),
                                             Eigen::MatrixXd::Ones(1, 1)),
                    std::invalid_argument);
  prediction(0, 0) = std::numeric_limits<double>::infinity();
  REQUIRE_THROWS_AS(kittenml::metrics::r2_score(truth, prediction), std::invalid_argument);
  REQUIRE_THROWS_AS(kittenml::metrics::mean_squared_error(truth, prediction), std::invalid_argument);
}
