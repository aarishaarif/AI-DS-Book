---
title: "Module 4: Machine Learning"
sidebar_position: 4
---

# Module 4: Machine Learning

**Level:** Intermediate | **Prerequisites:** Modules 1–3

> Machine learning gives computers the ability to learn from data without being explicitly programmed. This module covers the full supervised learning workflow, key algorithms, and model evaluation.

---

## Learning Objectives

- Distinguish supervised, unsupervised, and reinforcement learning
- Build and evaluate regression and classification models with scikit-learn
- Apply the full ML pipeline: preprocessing → training → evaluation → tuning
- Prevent overfitting using regularization, cross-validation, and proper splits
- Interpret model performance with appropriate metrics

## The Machine Learning Workflow

A model learns a mapping from features to a target by minimizing error on training data. A reliable workflow keeps the test set untouched until final evaluation: define the question, inspect and split data, fit preprocessing on training data, train candidates, validate choices, evaluate once, and monitor after deployment.

| Term | Meaning |
| --- | --- |
| Feature | An input variable used by a model |
| Target | The value the model predicts |
| Generalization | Performance on new data from the same process |
| Leakage | Evaluation information accidentally used during training |
| Baseline | A simple reference model for comparison |

The most common source of misleadingly good results is leakage. A scaler, imputer, or feature selection step must learn its parameters from the training split only.

---

## 4.1 What Is Machine Learning?

```
Traditional programming: Rules + Data → Output
Machine Learning:        Data + Output → Rules (learned)
```

### Types of ML

| Type | Description | Examples |
|---|---|---|
| **Supervised** | Labeled training data | Classification, Regression |
| **Unsupervised** | No labels — find structure | Clustering, Dimensionality Reduction |
| **Reinforcement** | Agent learns via rewards | Game playing, Robotics |

---

## 4.2 The ML Workflow

```
1. Define the problem
2. Collect and explore data (EDA)
3. Preprocess and feature engineering
4. Split data (train/validation/test)
5. Choose and train a model
6. Evaluate performance
7. Tune hyperparameters
8. Final evaluation on test set
9. Deploy
```

---

## 4.3 Data Preprocessing

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Load data
df = pd.read_csv('data.csv')

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Scale features (important for distance-based algorithms)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

> ⚠️ **Critical rule:** Fit the scaler **only** on training data, then transform both train and test. Fitting on the full dataset is **data leakage**.

---

## 4.4 Linear Regression

Predicts a continuous output by fitting a line (or hyperplane) through the data.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Generate sample data
np.random.seed(42)
X_train = np.random.rand(100, 1) * 10
y_train = 2 * X_train.squeeze() + 1 + np.random.randn(100)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_train)
print(f"R²: {r2_score(y_train, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_pred)):.3f}")
print(f"Coefficient: {model.coef_[0]:.3f}, Intercept: {model.intercept_:.3f}")
```

### Ridge and Lasso Regression (Regularization)

```python
from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=1.0)   # L2 regularization — shrinks all coefficients
lasso = Lasso(alpha=0.1)   # L1 regularization — can zero out coefficients

ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)
```

---

## 4.5 Classification

### Logistic Regression

Despite the name, it's a classification algorithm that outputs probabilities.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

### K-Nearest Neighbors (KNN)

KNN predicts a label by finding the *k* training examples closest to a new point. For classification it uses the majority label among those neighbors; for regression it averages their target values. “Training” mostly means saving the examples, so prediction can be expensive on a large dataset.

Distance depends on feature scale. If one feature is measured in thousands and another between zero and one, the large-scale feature can dominate the neighbor search. Fit scaling only on the training split, then reuse it for validation and test data. A scikit-learn pipeline keeps those steps together:

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5, weights="distance", metric="minkowski"),
)
knn.fit(X_train, y_train)
print(f"Test accuracy: {knn.score(X_test, y_test):.3f}")
```

Here `n_neighbors=5` uses five nearby rows. `weights="distance"` gives closer rows more influence than distant ones. The default Minkowski distance is Euclidean when `p=2`; `p=1` gives Manhattan distance. Choose *k*, the distance measure, and weighting with cross-validation rather than the test set. A very small *k* is sensitive to noise; a very large *k* smooths away local patterns. KNN is a useful, simple baseline for small datasets, but it can be slow and memory-heavy at prediction time and becomes less meaningful with many irrelevant features.

### Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

plt.figure(figsize=(12, 5))
plot_tree(tree, feature_names=iris.feature_names,
          class_names=iris.target_names, filled=True)
plt.show()
```

### Random Forest

An **ensemble** of decision trees that averages their predictions to reduce variance.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.3f}")

# Feature importance
importances = pd.Series(rf.feature_importances_, index=iris.feature_names)
importances.sort_values().plot(kind='barh')
plt.title('Feature Importances'); plt.show()
```

### Support Vector Machine (SVM)

```python
from sklearn.svm import SVC

svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train, y_train)
print(f"Accuracy: {accuracy_score(y_test, svm.predict(X_test)):.3f}")
```

---

## 4.6 Model Evaluation

### Classification Metrics

```python
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve
)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=iris.target_names).plot()
plt.show()
```

| Metric | Formula | Best for |
|---|---|---|
| Accuracy | (TP+TN)/Total | Balanced classes |
| Precision | TP/(TP+FP) | Minimize false positives |
| Recall | TP/(TP+FN) | Minimize false negatives |
| F1-Score | 2·P·R/(P+R) | Imbalanced classes |
| ROC-AUC | Area under ROC curve | Overall ranking ability |

### Regression Metrics

| Metric | Notes |
|---|---|
| MAE | Mean Absolute Error — interpretable units |
| RMSE | Root Mean Squared Error — penalizes large errors |
| R² | Proportion of variance explained (1 = perfect) |

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(rf, iris.data, iris.target, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

## 4.7 Unsupervised Learning

### K-Means Clustering

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='x', s=200, linewidth=3)
plt.title('K-Means Clustering'); plt.show()
```

### PCA — Dimensionality Reduction

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_2d = pca.fit_transform(iris.data)

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=iris.target, cmap='Set1')
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.title('PCA of Iris Dataset'); plt.show()
```

---

## 4.8 Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
```

---

## 4.9 Overfitting and Underfitting

```
Underfitting → model too simple → high bias → increase complexity
Overfitting  → model too complex → high variance → regularize or get more data
```

**Detecting overfitting:**
```python
train_score = model.score(X_train, y_train)
test_score  = model.score(X_test,  y_test)

if train_score - test_score > 0.1:
    print("Possible overfitting — gap is large")
```

**Solutions:**
- More training data
- Regularization (Ridge, Lasso, dropout)
- Reduce model complexity
- Cross-validation for model selection

---

## Common Mistakes

- Leaking test data into preprocessing (fitting scaler on full dataset)
- Using accuracy alone on imbalanced datasets
- Not scaling features for SVM, KNN, or logistic regression
- Ignoring class imbalance — always check `y.value_counts()`

---

## Practice Exercises

1. Train a Random Forest on the breast cancer dataset and plot feature importances.
2. Compare Logistic Regression, Decision Tree, and SVM on the same dataset using 5-fold CV.
3. Apply K-Means to the Iris dataset without using the labels; evaluate with silhouette score.
4. Use GridSearchCV to find the best hyperparameters for a Decision Tree.
5. Visualize the learning curve for a model to diagnose bias vs. variance.

---

## Review Questions

1. How do training, validation, and test sets serve different purposes?
2. Why must preprocessing be fitted on training data inside a pipeline?
3. When is accuracy misleading, and which metric could help instead?
4. How do KNN, SVM, and decision trees make predictions differently?
5. What information does PCA preserve, and what information can it lose?

## Further Reading

- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Hands-On Machine Learning (Aurélien Géron)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
- [Introduction to Statistical Learning (free PDF)](https://www.statlearning.com/)
