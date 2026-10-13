#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <ops.hpp>

using namespace kittenml;

// Numerically estimates d(f(x).sum())/dx via central differences,
// perturbing one element of x at a time.
Eigen::MatrixXd
numerical_gradient(std::function<double(const Eigen::MatrixXd &)> f,
                   const Eigen::MatrixXd &x, double eps = 1e-5) {

  Eigen::MatrixXd grad(x.rows(), x.cols());

  for (int i = 0; i < x.rows(); ++i) {
    for (int j = 0; j < x.cols(); ++j) {
      Eigen::MatrixXd x_plus = x;
      x_plus(i, j) += eps;

      Eigen::MatrixXd x_minus = x;
      x_minus(i, j) -= eps;

      grad(i, j) = (f(x_plus) - f(x_minus)) / (2 * eps);
    }
  }

  return grad;
}

// Runs a binary op both ways — analytically (real backward()) and
// numerically (finite differences) — and checks they agree.
template <typename OpFn>
void check_gradient(OpFn op, const Eigen::MatrixXd &a_data,
                    const Eigen::MatrixXd &b_data) {
  // Analytical
  auto a = std::make_shared<Tensor>(a_data);
  auto b = std::make_shared<Tensor>(b_data);
  auto out = op(a, b);
  out->backward();

  // Numerical, w.r.t. a (b held fixed)
  auto loss_wrt_a = [&](const Eigen::MatrixXd &a_perturbed) {
    auto a2 = std::make_shared<Tensor>(a_perturbed);
    auto b2 = std::make_shared<Tensor>(b_data);
    return op(a2, b2)->data.sum();
  };

  // Numerical, w.r.t. b (a held fixed)
  auto loss_wrt_b = [&](const Eigen::MatrixXd &b_perturbed) {
    auto a2 = std::make_shared<Tensor>(a_data);
    auto b2 = std::make_shared<Tensor>(b_perturbed);
    return op(a2, b2)->data.sum();
  };

  Eigen::MatrixXd expected_grad_a = numerical_gradient(loss_wrt_a, a_data);
  Eigen::MatrixXd expected_grad_b = numerical_gradient(loss_wrt_b, b_data);

  REQUIRE(a->grad.isApprox(expected_grad_a, 1e-4));
  REQUIRE(b->grad.isApprox(expected_grad_b, 1e-4));
}

template <typename OpFn>
void check_unary_gradient(OpFn op, const Eigen::MatrixXd &data,
                          const Eigen::MatrixXd &weights) {
  auto a = std::make_shared<Tensor>(data);
  auto w = std::make_shared<Tensor>(weights);
  auto loss = sum(op(a) * w);
  loss->backward();

  auto numerical_loss = [&](const Eigen::MatrixXd &perturbed) {
    auto a2 = std::make_shared<Tensor>(perturbed);
    return op(a2)->data.cwiseProduct(weights).sum();
  };

  Eigen::MatrixXd expected = numerical_gradient(numerical_loss, data);
  REQUIRE(a->grad.isApprox(expected, 1e-4));
}

TEST_CASE("gradient check: addition", "[gradcheck]") {
  Eigen::MatrixXd a_data(2, 2);
  a_data << 1.0, -2.0, 3.5, 0.5;
  Eigen::MatrixXd b_data(2, 2);
  b_data << 0.3, 1.2, -0.7, 2.0;
  check_gradient([](auto a, auto b) { return a + b; }, a_data, b_data);
}

TEST_CASE("gradient check: subtraction", "[gradcheck]") {
  Eigen::MatrixXd a_data(2, 2);
  a_data << 4.0, -1.0, 2.0, 0.5;
  Eigen::MatrixXd b_data(2, 2);
  b_data << 1.5, 0.5, -2.0, 3.0;
  check_gradient([](auto a, auto b) { return a - b; }, a_data, b_data);
}

TEST_CASE("gradient check: elementwise multiplication", "[gradcheck]") {
  Eigen::MatrixXd a_data(2, 2);
  a_data << 2.0, 3.0, -1.0, 0.5;
  Eigen::MatrixXd b_data(2, 2);
  b_data << 1.0, -2.0, 4.0, 3.0;
  check_gradient([](auto a, auto b) { return a * b; }, a_data, b_data);
}

TEST_CASE("gradient check: division", "[gradcheck]") {
  Eigen::MatrixXd a_data(2, 2);
  a_data << 4.0, 9.0, -2.0, 6.0;
  // kept well away from zero — division blows up numerically near it
  Eigen::MatrixXd b_data(2, 2);
  b_data << 2.0, 3.0, 1.0, 4.0;
  check_gradient([](auto a, auto b) { return a / b; }, a_data, b_data);
}

TEST_CASE("gradient check: matmul", "[gradcheck]") {
  // Deliberately non-square (2x3 * 3x4) — shape bugs can hide
  // silently on square matrices, but not on these.
  Eigen::MatrixXd a_data(2, 3);
  a_data << 1, 2, 3, 4, 5, 6;

  Eigen::MatrixXd b_data(3, 4);
  b_data << 1, 0, 2, 1, -1, 3, 0, 2, 2, 1, 1, 0;

  check_gradient([](auto a, auto b) { return matmul(a, b); }, a_data, b_data);
}

TEST_CASE("gradient check: diamond dependency (reused tensor)", "[gradcheck]") {
  // a feeds into two branches that later recombine — this is exactly
  // the case naive recursive backward gets wrong, and topo-sort fixes.
  Eigen::MatrixXd a_data(2, 2);
  a_data << 1.0, 2.0, 3.0, 4.0;
  Eigen::MatrixXd b_data(2, 2);
  b_data << 0.5, 1.5, 2.0, 1.0;
  Eigen::MatrixXd c_data(2, 2);
  c_data << 2.0, 0.5, 1.0, 3.0;

  auto a = std::make_shared<Tensor>(a_data);
  auto b = std::make_shared<Tensor>(b_data);
  auto c = std::make_shared<Tensor>(c_data);

  auto d = matmul(a, b);
  auto e = matmul(a, c);
  auto f = d + e;
  f->backward();

  // a's total gradient should be the SUM of its contribution through
  // both branches: d(f)/da via d, plus d(f)/da via e.
  Eigen::MatrixXd expected = Eigen::MatrixXd::Ones(2, 2) * b_data.transpose() +
                             Eigen::MatrixXd::Ones(2, 2) * c_data.transpose();

  REQUIRE(a->grad.isApprox(expected, 1e-8));
}

TEST_CASE("gradient check: sum", "[gradcheck]") {
  Eigen::MatrixXd data(2, 3);
  data << 1.0, -2.0, 3.0, 0.5, 4.0, -1.0;

  check_unary_gradient([](auto a) { return sum(a); }, data,
                       Eigen::MatrixXd::Constant(1, 1, 2.5));
  check_unary_gradient(
      [](auto a) { return sum(a, 0); }, data,
      (Eigen::MatrixXd(1, 3) << 1.0, -2.0, 0.5).finished());
  check_unary_gradient(
      [](auto a) { return sum(a, 1); }, data,
      (Eigen::MatrixXd(2, 1) << 1.5, -0.5).finished());
}

TEST_CASE("gradient check: sigmoid", "[gradcheck]") {
  Eigen::MatrixXd data(2, 3);
  data << -3.0, -0.5, 0.0, 0.7, 2.0, 5.0;
  Eigen::MatrixXd weights(2, 3);
  weights << 0.5, -1.0, 2.0, 1.5, -0.2, 0.8;

  check_unary_gradient([](auto a) { return sigmoid(a); }, data, weights);
}

TEST_CASE("gradient check: log", "[gradcheck]") {
  Eigen::MatrixXd data(2, 3);
  data << 0.2, 0.5, 1.0, 1.5, 3.0, 8.0;
  Eigen::MatrixXd weights(2, 3);
  weights << 1.0, -0.5, 2.0, 0.3, -1.2, 0.7;

  check_unary_gradient([](auto a) { return log(a); }, data, weights);

  auto zero = std::make_shared<Tensor>(Eigen::MatrixXd::Zero(1, 1));
  auto result = log(zero);
  result->backward();
  REQUIRE(std::isfinite(result->data(0, 0)));
  REQUIRE(zero->grad(0, 0) == 0.0);
}

TEST_CASE("gradient check: exp", "[gradcheck]") {
  Eigen::MatrixXd data(2, 3);
  data << -2.0, -0.3, 0.0, 0.4, 1.2, 2.0;
  Eigen::MatrixXd weights(2, 3);
  weights << 0.2, -0.8, 1.5, 2.0, -0.4, 0.7;

  check_unary_gradient([](auto a) { return exp(a); }, data, weights);
}

TEST_CASE("gradient check: softmax", "[gradcheck]") {
  Eigen::MatrixXd data(2, 3);
  data << 1000.0, 1001.0, 999.0, -1000.0, -998.0, -999.0;
  Eigen::MatrixXd weights(2, 3);
  weights << 0.5, -1.0, 2.0, 1.5, -0.2, 0.8;

  check_unary_gradient([](auto a) { return softmax(a); }, data, weights);
  check_unary_gradient([](auto a) { return softmax(a, 0); }, data, weights);
}
