# kitten-ml

Kitten-ML is a small, header-only, sklearn-flavored ML library in C++, with
an autograd engine at its core.

The goal is one coherent package for classical machine learning, neural
networks, preprocessing, data handling, and evaluation.

## Public headers

All supported headers live below `include/kittenml` and mirror their C++
namespaces:

```text
include/kittenml/
├── classical/       # k-NN, K-Means, PCA, and future estimators
├── data/            # datasets, loaders, and generators
├── detail/          # internal validation and distance helpers
├── metrics/         # classification and regression metrics
├── nn/              # layers, losses, and optimizers
├── classical.hpp    # classical ML umbrella header
├── data.hpp         # data umbrella header
├── metrics.hpp      # metrics umbrella header
├── nn.hpp           # neural-network umbrella header
└── kittenml.hpp     # complete public API
```

Prefer namespaced includes in new code:

```cpp
#include <kittenml/classical/knn.hpp>

kittenml::classical::KNNClassifier model(3);
model.fit(X_train, y_train);
Eigen::VectorXi predictions = model.predict(X_test);
```

## Prerequisites

    brew install cmake eigen catch2

## Build & run

    mkdir build && cd build
    cmake ..
    make
    ./demo
    ./knn_demo
    ./kmeans_demo
    ./pca_demo
    ctest --output-on-failure

The original `demo` trains a small MLP on XOR. The three classical demos show
k-NN classification, K-Means clustering, and PCA projection.

## Roadmap

| Step | Milestone                                                                    | Deliverable                                            | Status      |
| ---- | ---------------------------------------------------------------------------- | ------------------------------------------------------ | ----------- |
| 1    | Autograd core — `Tensor`, `+`, `*`, `matmul`, `backward()`                   | Gradients match hand-checked values                    | Started     |
| 2    | Ops + `nn` module — activations, losses, SGD                                 | MLP trains on a toy dataset (Iris/XOR)                 | Started     |
| 3    | Classical ML — kNN, K-Means, PCA, Decision Trees, Linear/Logistic Regression | Each has `fit()`/`predict()` or `transform()` + demo   | In progress: k-NN, K-Means, PCA |
| 4    | Data utils, benchmarks vs. sklearn, docs                                     | README + comparison numbers (stretch: Python bindings) | Not started |
