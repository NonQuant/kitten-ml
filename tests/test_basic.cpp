#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <data.hpp>

TEST_CASE("Eigen matrix multiply works as expected", "[sanity]") {
    Eigen::MatrixXd a(2, 2);
    a << 1, 2,
         3, 4;

    Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(2, 2);
    Eigen::MatrixXd result = a * identity;

    REQUIRE(result(0, 0) == 1);
    REQUIRE(result(0, 1) == 2);
    REQUIRE(result(1, 0) == 3);
    REQUIRE(result(1, 1) == 4);
}

TEST_CASE("Dataset stores samples row-wise", "[data]") {
    Eigen::MatrixXd X(3, 2);
    X << 1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0;
    Eigen::MatrixXd y(3, 1);
    y << 0.0,
         1.0,
         0.0;

    kittenml::Dataset dataset(X, y);

    REQUIRE(dataset.size() == 3);
    REQUIRE(dataset.X.isApprox(X));
    REQUIRE(dataset.y.isApprox(y));
}

TEST_CASE("Dataset rejects mismatched sample counts", "[data]") {
    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(3, 2);
    Eigen::MatrixXd y = Eigen::MatrixXd::Zero(2, 1);

    REQUIRE_THROWS_AS(kittenml::Dataset(X, y), std::invalid_argument);
}
