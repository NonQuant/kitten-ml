#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>

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
