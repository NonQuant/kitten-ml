#include <catch2/catch_test_macros.hpp>
#include <data.hpp>

#include <algorithm>
#include <utility>
#include <vector>

namespace {

kittenml::Dataset make_dataset(Eigen::Index size) {
  Eigen::MatrixXd X(size, 2);
  Eigen::MatrixXd y(size, 1);
  for (Eigen::Index sample = 0; sample < size; ++sample) {
    X(sample, 0) = static_cast<double>(sample);
    X(sample, 1) = static_cast<double>(sample + 10);
    y(sample, 0) = static_cast<double>(sample);
  }
  return {std::move(X), std::move(y)};
}

std::vector<int> read_epoch(kittenml::DataLoader &loader) {
  std::vector<int> order;
  kittenml::Batch batch;
  while (loader.next(batch)) {
    for (Eigen::Index row = 0; row < batch.y->data.rows(); ++row)
      order.push_back(static_cast<int>(batch.y->data(row, 0)));
  }
  return order;
}

} // namespace

TEST_CASE("DataLoader creates Tensor batches and keeps samples aligned",
          "[data][loader]") {
  kittenml::DataLoader loader(make_dataset(5), 2, false);
  kittenml::Batch batch;

  REQUIRE(loader.next(batch));
  REQUIRE(batch.X);
  REQUIRE(batch.y);
  REQUIRE(batch.X->data.rows() == 2);
  REQUIRE(batch.X->data.cols() == 2);
  REQUIRE(batch.y->data.rows() == 2);
  REQUIRE(batch.y->data.cols() == 1);
  REQUIRE(batch.X->data.col(0).isApprox(batch.y->data.col(0)));
  REQUIRE(batch.X->parents.empty());
  REQUIRE(batch.y->parents.empty());

  REQUIRE(loader.next(batch));
  REQUIRE(batch.X->data.rows() == 2);
  REQUIRE(batch.X->data.col(0).isApprox(batch.y->data.col(0)));

  REQUIRE(loader.next(batch));
  REQUIRE(batch.X->data.rows() == 1);
  REQUIRE(batch.y->data.rows() == 1);
  REQUIRE(batch.X->data(0, 0) == 4.0);
  REQUIRE_FALSE(loader.next(batch));
}

TEST_CASE("DataLoader reset starts a new epoch", "[data][loader]") {
  kittenml::DataLoader loader(make_dataset(5), 3, false);

  REQUIRE(read_epoch(loader) == std::vector<int>{0, 1, 2, 3, 4});
  REQUIRE(read_epoch(loader).empty());

  loader.reset();

  REQUIRE(read_epoch(loader) == std::vector<int>{0, 1, 2, 3, 4});
}

TEST_CASE("DataLoader shuffling is seeded and reproducible",
          "[data][loader]") {
  kittenml::DataLoader first(make_dataset(10), 4, true, 42);
  kittenml::DataLoader second(make_dataset(10), 4, true, 42);
  const std::vector<int> sequential{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  const std::vector<int> first_epoch = read_epoch(first);
  const std::vector<int> matching_first_epoch = read_epoch(second);
  REQUIRE(first_epoch == matching_first_epoch);
  REQUIRE(first_epoch != sequential);

  std::vector<int> sorted = first_epoch;
  std::sort(sorted.begin(), sorted.end());
  REQUIRE(sorted == sequential);

  first.reset();
  second.reset();
  const std::vector<int> second_epoch = read_epoch(first);
  REQUIRE(second_epoch == read_epoch(second));
  REQUIRE(second_epoch != first_epoch);
}

TEST_CASE("DataLoader validates batch size", "[data][loader]") {
  REQUIRE_THROWS_AS(kittenml::DataLoader(make_dataset(4), 0, false),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(kittenml::DataLoader(make_dataset(4), -1, false),
                    std::invalid_argument);
}
