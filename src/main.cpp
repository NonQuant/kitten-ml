#include <cmath>
#include <iostream>
#include <limits>

#include <kittenml/data.hpp>
#include <kittenml/nn.hpp>
#include <kittenml/ops.hpp>

using namespace kittenml;

int main() {
  constexpr int steps = 200;

  DataLoader loader(make_xor(), 4, true, 42);
  nn::Linear hidden_layer(2, 4);
  nn::Linear output_layer(4, 1);
  nn::MSELoss loss_function;
  nn::SGD optimizer({hidden_layer.W, hidden_layer.b, output_layer.W,
                     output_layer.b},
                    0.1);

  double initial_loss = std::numeric_limits<double>::quiet_NaN();
  double final_loss = std::numeric_limits<double>::quiet_NaN();

  for (int step = 1; step <= steps; ++step) {
    loader.reset();
    Batch batch;
    if (!loader.next(batch)) {
      std::cerr << "Failed to load the XOR batch\n";
      return 1;
    }

    const auto hidden = relu(hidden_layer.forward(batch.X));
    const auto prediction = output_layer.forward(hidden);
    const auto loss = loss_function(prediction, batch.y);

    final_loss = loss->data(0, 0);
    if (!std::isfinite(final_loss)) {
      std::cerr << "Training failed: loss is not finite at step " << step
                << '\n';
      return 1;
    }

    optimizer.zero_grad();
    loss->backward();
    optimizer.step();

    if (step == 1)
      initial_loss = final_loss;
    std::cout << "step " << step << "/" << steps << " loss: " << final_loss
              << '\n';
  }

  if (final_loss >= initial_loss * 0.5) {
    std::cerr << "Training failed: loss did not decrease enough from "
              << initial_loss << " to " << final_loss << '\n';
    return 1;
  }

  std::cout << "Loss decreased from " << initial_loss << " to " << final_loss
            << '\n';
  return 0;
}
