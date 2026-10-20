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

TEST_CASE("make_xor returns the four XOR samples", "[data]") {
    const kittenml::Dataset dataset = kittenml::make_xor();

    Eigen::MatrixXd expected_X(4, 2);
    expected_X << 0.0, 0.0,
                  0.0, 1.0,
                  1.0, 0.0,
                  1.0, 1.0;
    Eigen::MatrixXd expected_y(4, 1);
    expected_y << 0.0,
                  1.0,
                  1.0,
                  0.0;

    REQUIRE(dataset.X.isApprox(expected_X));
    REQUIRE(dataset.y.isApprox(expected_y));
}

TEST_CASE("make_blobs returns balanced reproducible classification data",
          "[data]") {
    const kittenml::Dataset first = kittenml::make_blobs(20, 3, 42);
    const kittenml::Dataset second = kittenml::make_blobs(20, 3, 42);

    REQUIRE(first.X.rows() == 20);
    REQUIRE(first.X.cols() == 2);
    REQUIRE(first.y.rows() == 20);
    REQUIRE(first.y.cols() == 1);
    REQUIRE(first.X == second.X);
    REQUIRE(first.y == second.y);

    Eigen::Vector3i class_counts = Eigen::Vector3i::Zero();
    for (Eigen::Index sample = 0; sample < first.size(); ++sample) {
        const double label = first.y(sample, 0);
        REQUIRE(label >= 0.0);
        REQUIRE(label < 3.0);
        REQUIRE(label == static_cast<Eigen::Index>(label));
        ++class_counts(static_cast<Eigen::Index>(label));
    }

    REQUIRE(class_counts.maxCoeff() - class_counts.minCoeff() <= 1);
}

TEST_CASE("make_blobs validates its dimensions", "[data]") {
    REQUIRE_THROWS_AS(kittenml::make_blobs(-1, 2, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(kittenml::make_blobs(10, 0, 0), std::invalid_argument);
}
