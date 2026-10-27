# kitten-ml

Kitten-ML: a small, sklearn-flavored ML library in C++, with an autograd engine at its core.

Larger idea is to have a single, unified library that combines pytorch, sklearn and other tools. Users should be able to do everything starting from preprocessing, model training, evaluation and deployment (idk about this tho).

## Prerequisites

    brew install cmake eigen catch2

## Build & run

    mkdir build && cd build
    cmake ..
    make
    ./demo
    ./tests

The demo trains a small MLP on XOR and prints the loss at every step. It exits
successfully after confirming that the loss is finite and decreases by at least
half. The test executable should report that all tests passed.

## Roadmap

| Step | Milestone                                                                    | Deliverable                                            | Status      |
| ---- | ---------------------------------------------------------------------------- | ------------------------------------------------------ | ----------- |
| 1    | Autograd core — `Tensor`, `+`, `*`, `matmul`, `backward()`                   | Gradients match hand-checked values                    | Started     |
| 2    | Ops + `nn` module — activations, losses, SGD                                 | MLP trains on a toy dataset (Iris/XOR)                 | Started     |
| 3    | Classical ML — kNN, K-Means, PCA, Decision Trees, Linear/Logistic Regression | Each has `fit()`/`predict()` + demo                    | Not started |
| 4    | Data utils, benchmarks vs. sklearn, docs                                     | README + comparison numbers (stretch: Python bindings) | Not started |
