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
├── classical/       # classifiers, regression, clustering, and PCA
├── data/            # datasets, loaders, and generators
├── detail/          # internal validation and distance helpers
├── metrics/         # classification and regression metrics
├── nn/              # layers, losses, and optimizers
├── preprocessing/   # StandardScaler and future feature transforms
├── classical.hpp    # classical ML umbrella header
├── data.hpp         # data umbrella header
├── metrics.hpp      # metrics umbrella header
├── nn.hpp           # neural-network umbrella header
├── preprocessing.hpp # preprocessing umbrella header
└── kittenml.hpp     # complete public API
```

Prefer namespaced includes in new code:

```cpp
#include <kittenml/classical/knn.hpp>

kittenml::classical::KNNClassifier model(3);
model.fit(X_train, y_train);
Eigen::VectorXi predictions = model.predict(X_test);
```

The five predictive estimators share a method convention, without requiring a
virtual base class:

| Estimator | Training | Inference |
| --- | --- | --- |
| KNNClassifier | `fit(X, y)` | `predict(X)` |
| DecisionTreeClassifier | `fit(X, y)` | `predict(X)` |
| LinearRegression | `fit(X, y)` | `predict(X)` |
| LogisticRegression | `fit(X, y)` | `predict(X)`, `predict_proba(X)` |
| KMeans | `fit(X)` | `predict(X)` |

All `fit()` methods return the estimator by reference; inference methods are
const. PCA is the sixth algorithm and uses `fit(X)` / `transform(X)`, or
`fit_transform(X)`, because its output is a projection.

Inputs use rows for samples and columns for features. Classifiers accept
`Eigen::VectorXi` labels or a single-column, integer-valued `Eigen::MatrixXd`.
Classifier predictions are integer vectors. Linear regression accepts matrix
targets, including multiple output columns, and returns matrix predictions.

The decision tree uses Gini impurity, with configurable maximum depth and minimum
split size; majority-vote ties choose the smallest label. Logistic regression
supports two arbitrary integer class labels. Its `predict_proba()` returns one
probability per sample for the larger label; `predict()` uses threshold 0.5.
Both regressions initialize weights and bias to zero and train with Tensor
autograd and full-batch SGD for the configured number of iterations.
Scale features before training and reduce the learning rate if descent diverges.

## Loading, splitting, and scaling data

```cpp
#include <kittenml/kittenml.hpp>

// Numeric CSV with a header; column 2 is the target (zero-based).
auto dataset = kittenml::data::load_csv("samples.csv", 2, {',', true});
auto split = kittenml::data::train_test_split(dataset, 0.2, true, 42);

kittenml::preprocessing::StandardScaler scaler;
auto X_train = scaler.fit_transform(split.X_train);
auto X_test = scaler.transform(split.X_test); // Reuses training statistics.

kittenml::classical::LinearRegression model;
auto predictions = model.fit(X_train, split.y_train).predict(X_test);
double mse = kittenml::metrics::mean_squared_error(split.y_test, predictions);
double r2 = kittenml::metrics::r2_score(split.y_test, predictions);
```

`data::read_csv(path, options)` returns an Eigen matrix without selecting a
target, and also accepts an input stream. Headers default to absent. Numeric
fields may be quoted; quoted headers may contain delimiters and escaped quotes.
Missing values, non-finite numbers, inconsistent row widths, malformed quoting,
and multiline quoted fields are rejected. Parse errors identify the line.
`load_csv()` requires an explicit target column and returns the existing
`kittenml::Dataset` type.

`train_test_split(X, y, test_size, shuffle, seed)` accepts matrix targets or
integer label vectors. It returns named `X_train`, `X_test`, `y_train`, and
`y_test` fields, preserving the target type. Test size is a fraction, rounded
up to a sample count. With shuffling disabled, the last samples form the test
partition; with shuffling enabled, the seed makes the split reproducible.
Neither partition may be empty. This is a random split without stratification.

`StandardScaler` uses per-feature population standard deviations. Constant
training columns use scale 1. `mean()` and `scale()` expose the learned
statistics; `transform()` never updates them.

## Evaluation metrics

Accuracy and confusion matrices accept integer label vectors:

```cpp
double accuracy = kittenml::metrics::accuracy(y_true, y_pred);
auto confusion = kittenml::metrics::confusion_matrix(y_true, y_pred);
// confusion.labels holds the sorted union of true and predicted labels.
// confusion.counts(i, j) counts true label i predicted as label j.
```

MSE averages squared errors over all samples and output columns. R² computes
a score for each output column and returns their uniform average; it requires
at least two samples. Constant targets score 1 for perfect predictions and 0
otherwise. Negative R² values are valid.

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
    ./decision_tree_demo
    ./linear_regression_demo
    ./logistic_regression_demo
    ctest --output-on-failure

The original `demo` trains a small MLP on XOR. Each classical algorithm has its
own demo.

## Roadmap

| Step | Milestone                                                                    | Deliverable                                            | Status      |
| ---- | ---------------------------------------------------------------------------- | ------------------------------------------------------ | ----------- |
| 1    | Autograd core — `Tensor`, `+`, `*`, `matmul`, `backward()`                   | Gradients match hand-checked values                    | Started     |
| 2    | Ops + `nn` module — activations, losses, SGD                                 | MLP trains on a toy dataset (Iris/XOR)                 | Started     |
| 3    | Classical ML — kNN, K-Means, PCA, Decision Trees, Linear/Logistic Regression | Each has `fit()`/`predict()` or `transform()` + demo   | Implemented |
| 4    | Data utils, benchmarks vs. sklearn, docs                                     | README + comparison numbers (stretch: Python bindings) | Not started |
