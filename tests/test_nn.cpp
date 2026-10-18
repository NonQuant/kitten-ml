#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
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

TEST_CASE("MSELoss computes the mean loss and gradients", "[nn][loss]") {
  Eigen::MatrixXd prediction_data(2, 2);
  prediction_data << 1.0, 2.0, 3.0, 5.0;
  Eigen::MatrixXd target_data(2, 2);
  target_data << 0.0, 2.0, 1.0, 4.0;
  auto prediction = std::make_shared<Tensor>(prediction_data);
  auto target = std::make_shared<Tensor>(target_data);

  auto loss = nn::MSELoss().forward(prediction, target);
  REQUIRE(loss->data(0, 0) == Catch::Approx(1.5));

  loss->backward();
  Eigen::MatrixXd expected_gradient(2, 2);
  expected_gradient << 0.5, 0.0, 1.0, 0.5;
  REQUIRE(prediction->grad.isApprox(expected_gradient));
  REQUIRE(target->grad.isApprox(-expected_gradient));
}

TEST_CASE("CrossEntropyLoss is stable and differentiates logits",
          "[nn][loss]") {
  Eigen::MatrixXd logits_data(2, 3);
  logits_data << 1000.0, 1001.0, 999.0, -1000.0, -998.0, -999.0;
  Eigen::MatrixXd target_data(2, 3);
  target_data << 0.0, 1.0, 0.0, 1.0, 0.0, 0.0;
  auto logits = std::make_shared<Tensor>(logits_data);
  auto target = std::make_shared<Tensor>(target_data);

  auto loss = nn::CrossEntropyLoss()(logits, target);
  REQUIRE(loss->data(0, 0) == Catch::Approx(1.4076059644));
  REQUIRE(std::isfinite(loss->data(0, 0)));

  loss->backward();
  Eigen::MatrixXd probabilities(2, 3);
  probabilities << 0.2447284711, 0.6652409558, 0.0900305732,
      0.0900305732, 0.6652409558, 0.2447284711;
  REQUIRE(logits->grad.isApprox((probabilities - target_data) / 2.0,
                                1e-9));
}

TEST_CASE("CrossEntropyLoss accepts class-index targets", "[nn][loss]") {
  Eigen::MatrixXd logits_data(2, 3);
  logits_data << 1.0, 2.0, 3.0, 3.0, 1.0, -1.0;
  Eigen::MatrixXd labels_data(2, 1);
  labels_data << 2.0, 0.0;
  auto logits = std::make_shared<Tensor>(logits_data);
  auto labels = std::make_shared<Tensor>(labels_data);

  auto loss = nn::CrossEntropyLoss().forward(logits, labels);
  REQUIRE(loss->data(0, 0) == Catch::Approx(0.2752685978));
  loss->backward();
  REQUIRE(logits->grad.rowwise().sum().isZero(1e-12));
  REQUIRE(labels->grad.isZero());
}
