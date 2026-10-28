#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <kittenml/classical/pca.hpp>

#include <cmath>

TEST_CASE("PCA projects onto the highest-variance direction",
          "[classical][pca]") {
  Eigen::MatrixXd X(5, 2);
  X << -2.0, -2.0, -1.0, -1.0, 0.0, 0.0,
        1.0,  1.0,  2.0,  2.0;

  kittenml::classical::PCA model(1);
  const Eigen::MatrixXd projected = model.fit_transform(X);

  REQUIRE(model.mean().isZero(1e-12));
  REQUIRE(model.covariance().isApprox(
      (Eigen::Matrix2d() << 2.5, 2.5, 2.5, 2.5).finished()));
  REQUIRE(std::abs(model.components()(0, 0)) ==
          Catch::Approx(1.0 / std::sqrt(2.0)));
  REQUIRE(model.explained_variance()(0) == Catch::Approx(5.0));
  REQUIRE(projected.rows() == X.rows());
  REQUIRE(projected.cols() == 1);
  REQUIRE(std::abs(projected.mean()) < 1e-12);
}

TEST_CASE("PCA rejects invalid component counts", "[classical][pca]") {
  REQUIRE_THROWS_AS(kittenml::classical::PCA(0), std::invalid_argument);
  kittenml::classical::PCA model(3);
  REQUIRE_THROWS_AS(model.fit(Eigen::MatrixXd::Ones(4, 2)),
                    std::invalid_argument);
}
