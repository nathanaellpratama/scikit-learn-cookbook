# scikit-learn Cookbook — Code Reproduction

This repository reproduces, explains, and extends the code from **_scikit-learn Cookbook, Third Edition_** (John Sukup, Packt Publishing / O'Reilly), Chapters 1–5. It was built as a machine learning coursework assignment: for every chapter, the book's recipes are reproduced in a Jupyter notebook, each technique is explained from first principles, and the results are verified numerically (not just illustrated) — often with extra experiments beyond what the book itself shows.

**Author:** Nathanael — S1 Teknik Komputer, Telkom University

## Repository Structure

```
scikit-learn-cookbook/
├── 01. API Elements of scikit-learn/
│   └── ch01_sklearn_api.ipynb
├── 02. Data Preprocessing/
│   └── ch02_preprocessing.ipynb
├── 03. Dimensionality Reduction/
│   └── Chapter_03_Dimensionality_Reduction.ipynb
├── 04.Models with Distance Metrics and Nearest Neighbors/
│   └── Chapter_04_Distance_Metrics_KNN.ipynb
├── 05. Linear Models and Regularization/
│   └── Chapter_05_Linear_Models_Regularization.ipynb
├── requirements.txt
└── README.md
```

Each notebook is fully self-contained and already executed, so every number, table, and plot in it is a genuine output of the code above it, not hand-typed.

## Chapter Summaries

### Chapter 1 — API Elements of scikit-learn
Covers the design philosophy behind scikit-learn's consistent API: estimators (`fit`/`predict`/`score`), transformers (`fit`/`transform`), and the `Pipeline`/`ColumnTransformer` composition pattern. Includes a hands-on data-leakage demonstration (fitting a transformer on train+test vs. train only), two custom estimators built from `BaseEstimator`/`TransformerMixin`/`ClassifierMixin` (an IQR-based outlier clipper and a nearest-centroid classifier), a leaky-vs-correct feature-selection-inside-CV comparison on pure noise data, hyperparameter search (`GridSearchCV`, `RandomizedSearchCV`, `HalvingGridSearchCV`), estimator tags, and a metadata-routing (`sample_weight`) example verified by comparing fitted coefficients.

### Chapter 2 — Data Preprocessing
Covers why raw data breaks naive models (missing values, unscaled features, outliers) through controlled experiments, then each preprocessing family: imputers (`SimpleImputer`/`KNNImputer`/`IterativeImputer`, compared via a synthetic RMSE benchmark on correlated vs. independent features), scalers (`StandardScaler`/`MinMaxScaler`/`Normalizer`/`RobustScaler`, with a model-sensitivity table across five classifiers), and encoders (`OneHotEncoder`/`OrdinalEncoder`/`LabelEncoder`/`TargetEncoder`). Builds leakage-safe pipelines with `GridSearchCV`-tuned preprocessing, feature engineering (`PolynomialFeatures`, `KBinsDiscretizer`, `FunctionTransformer`, feature selection methods), and a practical end-to-end regression exercise with diagnostic plots.

### Chapter 3 — Dimensionality Reduction
Demonstrates the curse of dimensionality (distance contrast collapse, KNN accuracy degradation with added noise features), then PCA in depth — including a from-scratch eigen-decomposition that is checked to match scikit-learn's implementation exactly, scree plots, loadings, reconstruction error, and why scaling is required before PCA. Covers LDA (Fisher's criterion, class-separability) and t-SNE (perplexity sensitivity, random-seed sensitivity, and an honest discussion of its interpretation caveats), closing with a direct comparison of how each method affects downstream model accuracy.

### Chapter 4 — Models with Distance Metrics and Nearest Neighbors
Builds up distance metrics (Euclidean, Manhattan, Chebyshev, Minkowski, cosine, Hamming) with hand-computed verification against SciPy/scikit-learn, then a from-scratch KNN implementation checked against `KNeighborsClassifier`. Explores the bias-variance effect of `k`, decision boundaries, search-algorithm benchmarking (`kd_tree`/`ball_tree`/`brute`), a robust multi-split metric comparison (replacing a misleading single-split comparison with a 200-repeat `ShuffleSplit` evaluation), nested cross-validation to expose optimistic bias in naive model selection, and imbalanced-data evaluation (oversampling, balanced accuracy, PR-AUC).

### Chapter 5 — Linear Models and Regularization
Starts from OLS and the normal equation (verified from scratch), then a multicollinearity case study with VIF, residual diagnostics, and adjusted R². Covers Ridge and Lasso (closed-form Ridge verification, why standardization is required, an alpha-sweep showing the classic U-shaped test-error curve, and `RidgeCV`/`LassoCV` tuning), ElasticNet (the grouping effect, coefficient paths via `enet_path`), a 200-repetition bias-variance simulation, polynomial regression and `SplineTransformer` for non-linear relationships, and a final set of practical exercises comparing Ridge/Lasso/ElasticNet across different synthetic data structures (dense-correlated, sparse, grouped-sparse) to show when each regularizer wins.

## How to Run

```bash
git clone <this-repository-url>
cd scikit-learn-cookbook
pip install -r requirements.txt
jupyter notebook
```

Open any chapter's notebook and run all cells top to bottom, or simply read it — every cell has already been executed and its output saved.

## Reference

Sukup, J. (2024). *scikit-learn Cookbook, Third Edition*. Packt Publishing / O'Reilly Media.
