#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <nn.hpp>

using namespace kittenml;

TEST_CASE("Linear forward and backward", "[nn]") {
  nn::Linear layer(3, 2);
  layer.W->data << 1.0, -1.0, 2.0, 0.5, -0.5, 3.0;
  layer.b->data << 0.25, -0.75;

  Eigen::MatrixXd x_data(2, 3);
  x_data << 1.0, 2.0, -1.0, 0.5, -2.0, 3.0;
  auto x = std::make_shared<Tensor>(x_data);
  auto out = layer.forward(x);

  Eigen::MatrixXd expected(2, 2);
  expected << 5.75, -3.75, -4.75, 6.75;
  REQUIRE(out->data.isApprox(expected));

  out->backward();

  Eigen::MatrixXd expected_x_grad(2, 3);
  expected_x_grad << 0.0, 2.5, 2.5, 0.0, 2.5, 2.5;
  Eigen::MatrixXd expected_W_grad(3, 2);
  expected_W_grad << 1.5, 1.5, 0.0, 0.0, 2.0, 2.0;

  REQUIRE(x->grad.isApprox(expected_x_grad));
  REQUIRE(layer.W->grad.isApprox(expected_W_grad));
  REQUIRE(layer.b->grad.isApprox(Eigen::MatrixXd::Constant(1, 2, 2.0)));
}
