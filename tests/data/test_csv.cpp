#include <kittenml/data/csv.hpp>
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <sstream>

TEST_CASE("CSV parses numeric rows, headers, quoting and delimiters", "[data][csv]") {
  std::istringstream input("\"first;column\";\"second\"\"column\"\r\n\"1\"; 2 \r\n3;4\r\n");
  const auto matrix = kittenml::data::read_csv(input, {';', true});
  Eigen::Matrix2d expected;
  expected << 1, 2, 3, 4;
  REQUIRE(matrix.isApprox(expected));
  std::istringstream no_header("1,2\n3,4");
  REQUIRE(kittenml::data::read_csv(no_header).isApprox(expected));
}

TEST_CASE("CSV loads files and explicitly selects a target column", "[data][csv]") {
  const auto path = std::filesystem::path(__FILE__).parent_path() /
                    "fixtures/numeric.csv";
  const auto dataset = kittenml::data::load_csv(path.string(), 1, {',', true});
  REQUIRE(dataset.X.rows() == 3);
  REQUIRE(dataset.X.cols() == 2);
  REQUIRE(dataset.X(2, 1) == 30);
  REQUIRE(dataset.y(1, 0) == 1);
  REQUIRE(kittenml::data::load_csv(path.string(), 0, {',', true}).X(0, 0) == 0);
  REQUIRE(kittenml::data::load_csv(path.string(), 2, {',', true}).y(0, 0) == 10);
  REQUIRE_THROWS_AS(kittenml::data::load_csv(path.string(), 3, {',', true}),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(kittenml::data::read_csv(path.string() + ".missing"),
                    std::runtime_error);
}

TEST_CASE("CSV rejects missing, malformed and nonnumeric data", "[data][csv]") {
  for (const auto *text : {"", "1,2\n3", "1,\n", "1,nan", "1,inf",
                           "1,2junk", "1,\"2", "1,2\"3", "1,\"2\"x",
                           "1,2\n\n"}) {
    std::istringstream input(text);
    REQUIRE_THROWS_AS(kittenml::data::read_csv(input), std::invalid_argument);
  }
  std::istringstream header_only("a,b\n");
  REQUIRE_THROWS_AS(kittenml::data::read_csv(header_only, {',', true}),
                    std::invalid_argument);
}
