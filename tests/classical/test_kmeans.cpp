#include <catch2/catch_test_macros.hpp>
#include <kittenml/classical/kmeans.hpp>

TEST_CASE("K-Means finds separated cluster centroids",
          "[classical][kmeans]") {
  Eigen::MatrixXd X(6, 2);
  X << -5.0, -5.0, -5.0, -4.0, -4.0, -5.0,
        5.0,  5.0,  5.0,  4.0,  4.0,  5.0;

  kittenml::classical::KMeans model(2, 100, 1e-10, 8);
  model.fit(X);

  REQUIRE(model.n_iterations() <= 100);
  REQUIRE(model.inertia() < 3.0);
  REQUIRE((model.labels().head(3).array() == model.labels()(0)).all());
  REQUIRE((model.labels().tail(3).array() == model.labels()(3)).all());
  REQUIRE(model.labels()(0) != model.labels()(3));

  Eigen::MatrixXd queries(2, 2);
  queries << -4.8, -4.8, 4.8, 4.8;
  const Eigen::VectorXi predictions = model.predict(queries);
  REQUIRE(predictions(0) != predictions(1));
}

TEST_CASE("K-Means is reproducible for a fixed seed",
          "[classical][kmeans]") {
  Eigen::MatrixXd X = Eigen::MatrixXd::Random(20, 3);
  kittenml::classical::KMeans first(3, 50, 1e-6, 42);
  kittenml::classical::KMeans second(3, 50, 1e-6, 42);
  first.fit(X);
  second.fit(X);
  REQUIRE(first.centroids().isApprox(second.centroids()));
  REQUIRE(first.labels() == second.labels());
}
