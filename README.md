# kitten-ml

Kitten-ML: a small, sklearn-flavored ML library in C++, with an autograd engine at its core.

## Prerequisites

    brew install cmake eigen catch2

## Build & run

    mkdir build && cd build
    cmake ..
    make
    ./demo
    ./tests

If `demo` prints a 2x2 matrix and its transpose, and `tests` reports
all tests passed, your environment is ready.

## Roadmap

| Step | Milestone                                                                    | Deliverable                                            |
| ---- | ---------------------------------------------------------------------------- | ------------------------------------------------------ |
| 1    | Autograd core — `Tensor`, `+`, `*`, `matmul`, `backward()`                   | Gradients match hand-checked values                    |
| 2    | Ops + `nn` module — activations, losses, SGD                                 | MLP trains on a toy dataset (Iris/XOR)                 |
| 3    | Classical ML — kNN, K-Means, PCA, Decision Trees, Linear/Logistic Regression | Each has `fit()`/`predict()` + demo                    |
| 4    | Data utils, benchmarks vs. sklearn, docs                                     | README + comparison numbers (stretch: Python bindings) |
