---
title: "Module 3: Statistics and Mathematics for Machine Learning"
sidebar_position: 3
---

# Module 3: Statistics and Mathematics for Machine Learning

**Level:** Beginner–Intermediate | **Prerequisites:** Module 2 (Python for Data Science)

> Machine learning is applied mathematics. This module builds the statistical and mathematical intuition you need to understand why algorithms work — and when they fail.

---

## Learning Objectives

- Describe data using descriptive statistics and distributions
- Apply probability theory to model uncertainty
- Use linear algebra for data representation and transformations
- Apply calculus concepts to understand gradient-based optimization
- Select appropriate statistical tests for hypothesis testing

---

## 3.1 Descriptive Statistics

### Measures of Central Tendency

```python
import numpy as np

data = [12, 15, 14, 10, 18, 14, 16, 13, 14, 20]

mean   = np.mean(data)    # 14.6
median = np.median(data)  # 14.0
# Mode: most frequent value
from scipy import stats
mode = stats.mode(data).mode   # 14
```

**Mean** — sensitive to outliers.  
**Median** — robust to outliers; use for skewed data.  
**Mode** — the most common value; useful for categorical data.

### Measures of Spread

```python
variance = np.var(data, ddof=1)      # 7.38 (sample variance)
std_dev  = np.std(data, ddof=1)      # 2.72
iqr      = np.percentile(data, 75) - np.percentile(data, 25)  # IQR
```

### Skewness and Kurtosis

```python
from scipy.stats import skew, kurtosis

print(skew(data))      # > 0: right-skewed, < 0: left-skewed
print(kurtosis(data))  # > 0: heavy tails (leptokurtic)
```

---

## 3.2 Probability

### Basic Rules

| Rule | Formula |
|---|---|
| Addition (mutually exclusive) | P(A ∪ B) = P(A) + P(B) |
| Multiplication (independent) | P(A ∩ B) = P(A) × P(B) |
| Conditional probability | P(A \| B) = P(A ∩ B) / P(B) |

### Bayes' Theorem

```
P(H | E) = P(E | H) × P(H) / P(E)
```

- **Prior** P(H): belief before seeing evidence
- **Likelihood** P(E | H): how probable the evidence is given the hypothesis
- **Posterior** P(H | E): updated belief after evidence

**Example (spam filter):**
If 30% of emails are spam, and 80% of spam contains "free" while only 5% of non-spam does:

```python
p_spam  = 0.30
p_free_given_spam    = 0.80
p_free_given_no_spam = 0.05

p_free = p_free_given_spam * p_spam + p_free_given_no_spam * (1 - p_spam)
p_spam_given_free = (p_free_given_spam * p_spam) / p_free
print(f"{p_spam_given_free:.2%}")  # 82.76%
```

---

## 3.3 Probability Distributions

### Normal (Gaussian) Distribution

The bell curve — the most important distribution in statistics.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mu, sigma = 0, 1
x = np.linspace(-4, 4, 200)
plt.plot(x, norm.pdf(x, mu, sigma))
plt.title('Standard Normal Distribution')
plt.xlabel('x'); plt.ylabel('Probability Density')
plt.show()
```

**68-95-99.7 rule:** 68% of data falls within ±1σ, 95% within ±2σ, 99.7% within ±3σ.

### Other Key Distributions

| Distribution | Use Case |
|---|---|
| Bernoulli | Single binary outcome |
| Binomial | Count of successes in n trials |
| Poisson | Count of events in fixed time/space |
| Uniform | Equal probability across range |
| Exponential | Time between events |

```python
from scipy.stats import binom, poisson

# Probability of exactly 3 heads in 10 coin flips
print(binom.pmf(3, n=10, p=0.5))   # 0.117

# Probability of seeing 2 events when average is 4
print(poisson.pmf(2, mu=4))         # 0.147
```

---

## 3.4 Linear Algebra

### Vectors and Matrices

```python
import numpy as np

v = np.array([1, 2, 3])         # vector
A = np.array([[1, 2], [3, 4]])   # matrix

# Operations
print(A.T)           # transpose
print(A @ A)         # matrix multiplication
print(np.linalg.det(A))          # determinant: -2
print(np.linalg.inv(A))          # inverse
```

### Dot Product

The dot product measures similarity between vectors:

```python
a = np.array([1, 0, 0])
b = np.array([0, 1, 0])
c = np.array([1, 0, 0])

print(np.dot(a, b))  # 0 — orthogonal (no similarity)
print(np.dot(a, c))  # 1 — identical direction
```

### Norms

```python
v = np.array([3, 4])
print(np.linalg.norm(v))          # L2 norm: 5.0
print(np.linalg.norm(v, ord=1))   # L1 norm: 7.0
```

### Eigenvalues and Eigenvectors

Critical for **PCA** (dimensionality reduction):

```python
A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print(eigenvalues)    # [5. 2.]
print(eigenvectors)
```

---

## 3.5 Calculus for ML

### Derivatives

A derivative measures how much a function changes as its input changes. In ML, we minimize a **loss function** by following the gradient downhill.

```python
# Numerical derivative
def f(x): return x**2 + 3*x + 1

def derivative(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h

print(derivative(f, 2.0))   # ≈ 7.0 (exact: 2x+3 = 7)
```

### Gradient Descent

The core optimization algorithm behind neural networks:

```python
def gradient_descent(f_grad, start, lr=0.1, n_steps=100):
    x = start
    for _ in range(n_steps):
        x = x - lr * f_grad(x)
    return x

# Minimize f(x) = x^2 — gradient is 2x
f_grad = lambda x: 2 * x
minimum = gradient_descent(f_grad, start=10.0)
print(f"Minimum at x ≈ {minimum:.4f}")  # ≈ 0.0
```

**Key concepts:**
- **Learning rate** (lr): step size — too large diverges, too small is slow
- **Gradient**: direction of steepest ascent; we negate it to descend
- **Local vs. global minimum**: gradient descent may get stuck in local minima

---

## 3.6 Hypothesis Testing

### Core Concepts

- **Null hypothesis (H₀):** No effect or difference exists
- **Alternative hypothesis (H₁):** An effect exists
- **p-value:** Probability of observing this result if H₀ is true
- **Significance level (α):** Threshold for rejecting H₀ (typically 0.05)

### t-Test

```python
from scipy.stats import ttest_ind
import numpy as np

group_a = np.random.normal(50, 10, 100)   # mean=50
group_b = np.random.normal(55, 10, 100)   # mean=55

t_stat, p_val = ttest_ind(group_a, group_b)
print(f"t={t_stat:.2f}, p={p_val:.4f}")

if p_val < 0.05:
    print("Reject H₀ — significant difference found")
else:
    print("Fail to reject H₀")
```

### Chi-Square Test

```python
from scipy.stats import chi2_contingency

# Observed: gender vs. product preference
observed = np.array([[30, 20], [15, 35]])
chi2, p, dof, expected = chi2_contingency(observed)
print(f"χ²={chi2:.2f}, p={p:.4f}")
```

---

## 3.7 Correlation and Covariance

```python
import pandas as pd, seaborn as sns

df = sns.load_dataset('tips')
numeric = df.select_dtypes('number')

# Pearson correlation matrix
corr = numeric.corr()
print(corr)

# Covariance matrix
cov = numeric.cov()
```

**Pearson r interpretation:**
- r ≈ 1: strong positive linear relationship
- r ≈ -1: strong negative linear relationship
- r ≈ 0: no linear relationship (may still have non-linear)

> ⚠️ **Correlation ≠ Causation.** Two variables can be correlated without one causing the other.

---

## Common Mistakes

- Applying mean/std to heavily skewed data without checking for outliers first
- Misinterpreting p-value as "the probability the null hypothesis is true"
- Forgetting to check distribution assumptions before choosing a statistical test
- Confusing variance with standard deviation in equations

---

## Practice Exercises

1. Generate 1000 samples from a normal distribution and verify the 68-95-99.7 rule empirically.
2. Implement Bayes' theorem to compute the probability of a disease given a positive test result.
3. Apply gradient descent to minimize `f(x) = (x-3)² + 2`. Plot the convergence.
4. Compute the correlation matrix for the Iris dataset and identify the most correlated features.
5. Run a two-sample t-test on two groups and interpret the result at α=0.05.

---

## Further Reading

- [Khan Academy — Statistics and Probability](https://www.khanacademy.org/math/statistics-probability)
- [3Blue1Brown — Essence of Linear Algebra (YouTube)](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [Think Stats (free book)](https://greenteapress.com/wp/think-stats-2e/)
