#pragma once

#include <kittenml/data/dataset.hpp>
#include <cmath>
#include <fstream>
#include <istream>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

namespace kittenml::data {

// Numeric CSV: optional header, quoted fields (including escaped quotes), and
// configurable delimiter. Missing/non-finite values and multiline fields fail.
struct CSVOptions {
  char delimiter = ',';
  bool has_header = false;
};

namespace csv_detail {
inline std::vector<std::string> fields(const std::string &line, char delimiter) {
  std::vector<std::string> result;
  std::string field;
  bool quoted = false, closed = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (quoted) {
      if (c == '"') {
        if (i + 1 < line.size() && line[i + 1] == '"') {
          field += '"';
          ++i;
        } else {
          quoted = false;
          closed = true;
        }
      } else field += c;
    } else if (c == delimiter) {
      result.push_back(field);
      field.clear();
      closed = false;
    } else if (c == '"' && field.empty() && !closed) {
      quoted = true;
    } else if (c == '"' || closed) {
      throw std::invalid_argument("malformed quoted field");
    } else field += c;
  }
  if (quoted) throw std::invalid_argument("unterminated quoted field");
  result.push_back(field);
  return result;
}
} // namespace csv_detail

inline Eigen::MatrixXd read_csv(std::istream &input, CSVOptions options = {}) {
  if (options.delimiter == '"' || options.delimiter == '\n' ||
      options.delimiter == '\r' || options.delimiter == '\0')
    throw std::invalid_argument("invalid CSV delimiter");
  std::string line;
  std::size_t line_number = 0, columns = 0, rows = 0;
  std::vector<double> values;
  while (std::getline(input, line)) {
    ++line_number;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    try {
      const auto fields = csv_detail::fields(line, options.delimiter);
      if (line_number == 1) columns = fields.size();
      if (fields.size() != columns)
        throw std::invalid_argument("inconsistent column count");
      if (line_number == 1 && options.has_header) continue;
      for (const auto &field : fields) {
        std::istringstream parser(field);
        parser.imbue(std::locale::classic());
        double value;
        if (!(parser >> value) || !std::isfinite(value))
          throw std::invalid_argument("missing or invalid numeric value");
        parser >> std::ws;
        if (!parser.eof())
          throw std::invalid_argument("invalid numeric value");
        values.push_back(value);
      }
      ++rows;
    } catch (const std::invalid_argument &error) {
      throw std::invalid_argument("CSV line " + std::to_string(line_number) +
                                  ": " + error.what());
    }
  }
  if (input.bad() || (input.fail() && !input.eof()))
    throw std::runtime_error("failed reading CSV stream");
  if (rows == 0) throw std::invalid_argument("CSV contains no data rows");
  Eigen::MatrixXd matrix(rows, columns);
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t column = 0; column < columns; ++column)
      matrix(row, column) = values[row * columns + column];
  return matrix;
}

inline Eigen::MatrixXd read_csv(const std::string &path, CSVOptions options = {}) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open CSV: " + path);
  return read_csv(input, options);
}

// target_column is zero-based; target selection is always explicit.
inline Dataset load_csv(const std::string &path, Eigen::Index target_column,
                        CSVOptions options = {}) {
  const Eigen::MatrixXd matrix = read_csv(path, options);
  if (matrix.cols() < 2 || target_column < 0 || target_column >= matrix.cols())
    throw std::invalid_argument("invalid CSV target column");
  Eigen::MatrixXd X(matrix.rows(), matrix.cols() - 1);
  X.leftCols(target_column) = matrix.leftCols(target_column);
  X.rightCols(matrix.cols() - target_column - 1) =
      matrix.rightCols(matrix.cols() - target_column - 1);
  return Dataset(std::move(X), matrix.col(target_column));
}

} // namespace kittenml::data
