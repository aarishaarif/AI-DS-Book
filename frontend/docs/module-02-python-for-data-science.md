---
title: "Module 2: Python for Data Science — NumPy, Pandas & Visualization"
sidebar_position: 2
---

# Module 2: Python for Data Science

**Level:** Beginner–Intermediate | **Prerequisites:** Module 1 (Python Foundations)

> This module covers the three pillars of the Python data science stack: **NumPy** for fast numerical computation, **Pandas** for tabular data manipulation, and **Matplotlib / Seaborn** for visualization.

---

## Learning Objectives

- Create and manipulate multi-dimensional NumPy arrays
- Load, clean, filter, and transform data with Pandas DataFrames
- Perform aggregation, grouping, and joining operations
- Build informative charts with Matplotlib and Seaborn
- Apply a full Exploratory Data Analysis (EDA) workflow

## The Data Science Mental Model

Data science turns observations into decisions through a repeatable loop: inspect the data, clean it, summarize it, visualize patterns, and question the result. NumPy provides efficient numerical arrays, Pandas adds labeled tables, and visualization makes distributions and relationships visible. A chart is evidence to investigate, not proof by itself.

| Term | Meaning |
| --- | --- |
| Vectorization | Applying an operation to an array without a Python loop |
| Index | The label or position used to select data |
| Missing value | An observation with no recorded value |
| Aggregation | Combining rows into a summary such as a mean or count |
| Distribution | The pattern of values and their frequencies |

---

## 2.1 NumPy — Numerical Python

NumPy's core data structure is the **ndarray** — a fast, memory-efficient n-dimensional array backed by C code.

```python
import numpy as np

# 1-D array
a = np.array([1, 2, 3, 4, 5])
print(a.shape)   # (5,)
print(a.dtype)   # int64

# 2-D array
mat = np.array([[1, 2, 3], [4, 5, 6]])
print(mat.shape) # (2, 3)

# Useful constructors
np.zeros((3, 4))          # all zeros
np.ones((2, 2))           # all ones
np.eye(3)                 # identity matrix
np.arange(0, 10, 2)       # [0 2 4 6 8]
np.linspace(0, 1, 5)      # [0. 0.25 0.5 0.75 1.]
np.random.rand(3, 3)      # uniform random
```

### Vectorized Operations

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)       # [5 7 9]
print(a * b)       # [4 10 18]
print(a ** 2)      # [1 4 9]
print(a + 10)      # broadcasting: [11 12 13]
```

:::tip How this code works
NumPy applies arithmetic element by element. In `a + 10`, broadcasting reuses the single value `10` across every element, so you can transform a whole array without writing a loop.
:::

### Indexing and Slicing

```python
mat = np.array([[1,2,3],[4,5,6],[7,8,9]])

print(mat[1, 2])      # 6
print(mat[:, 1])      # [2 5 8] — entire column 1
print(mat[0:2, 0:2])  # top-left 2×2
```

### Aggregation

```python
data = np.array([4, 7, 2, 9, 1])
print(np.mean(data))    # 4.6
print(np.std(data))     # 2.87
print(np.argmax(data))  # 3 — index of maximum
```

---

## 2.2 Pandas — Data Manipulation

```python
import pandas as pd

data = {
    'name':   ['Alice', 'Bob', 'Charlie'],
    'age':    [25, 30, 35],
    'salary': [55000, 72000, 90000],
    'dept':   ['Engineering', 'Marketing', 'Engineering'],
}
df = pd.DataFrame(data)
print(df)
```

**Output:**
```
      name  age  salary         dept
0    Alice   25   55000  Engineering
1      Bob   30   72000    Marketing
2  Charlie   35   90000  Engineering
```

### Loading Data

```python
df = pd.read_csv('data.csv')
print(df.shape)      # (rows, cols)
print(df.head())     # first 5 rows
print(df.info())     # dtypes + non-null counts
print(df.describe()) # summary statistics
```

### Selecting and Filtering

```python
# Select column
df['salary']

# Select rows with boolean mask
df[df['salary'] > 60000]

# .loc (label-based) and .iloc (position-based)
df.loc[0]           # first row by label
df.iloc[1, 2]       # row 1, col 2 by position

# query syntax
df.query("dept == 'Engineering' and age < 35")
```

### Cleaning

```python
df.isnull().sum()                       # count missing values
df.dropna()                             # drop rows with NaN
df.fillna(df['salary'].mean())          # fill NaN with mean
df.drop_duplicates(inplace=True)        # remove duplicates
df.rename(columns={'salary': 'pay'}, inplace=True)
```

### GroupBy and Aggregation

```python
summary = df.groupby('dept')['salary'].agg(['mean', 'count'])
print(summary)
#              mean  count
# dept
# Engineering  72500      2
# Marketing    72000      1
```

:::tip How this code works
`groupby('dept')` splits the table by department, then `agg` calculates two summaries for each group. This split–apply–combine pattern turns row-level data into a useful report.
:::

### Merging DataFrames

```python
employees = pd.DataFrame({'id': [1,2,3], 'name': ['A','B','C']})
projects  = pd.DataFrame({'id': [1,2,4], 'proj': ['X','Y','Z']})

pd.merge(employees, projects, on='id', how='inner')  # matching only
pd.merge(employees, projects, on='id', how='left')   # keep all employees
```

---

## 2.3 Visualization

### Matplotlib

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 100)
plt.figure(figsize=(8, 4))
plt.plot(x, np.sin(x), label='sin(x)', color='steelblue')
plt.title('Sine Wave')
plt.xlabel('x'); plt.ylabel('y')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()
```

### Seaborn

```python
import seaborn as sns

tips = sns.load_dataset('tips')

# Distribution
sns.histplot(tips['total_bill'], bins=20, kde=True)

# Box plot
sns.boxplot(data=tips, x='day', y='total_bill')

# Heatmap
corr = tips.select_dtypes('number').corr()
sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2f')
plt.show()
```

---

## 2.4 EDA Workflow Example

```python
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt

df = sns.load_dataset('titanic')

# 1. Shape and types
print(df.shape, df.dtypes)

# 2. Missing values
print(df.isnull().sum().sort_values(ascending=False))

# 3. Survival rate
print(df['survived'].value_counts(normalize=True))

# 4. Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df['age'].dropna(), bins=30, ax=axes[0], kde=True)
sns.countplot(data=df, x='pclass', hue='survived', ax=axes[1])
plt.tight_layout(); plt.show()
```

---

## Common Mistakes

- Using `df[col][row] = val` instead of `df.loc[row, col] = val` (chained assignment warning)
- Forgetting that most Pandas operations return a new DataFrame unless `inplace=True`
- Confusing `.loc` (labels) with `.iloc` (integer positions)
- Not calling `plt.show()` in scripts

---

## Practice Exercises

1. Create a 5×5 NumPy array of random integers; compute the row and column sums.
2. Load a CSV, identify the top 3 correlated numeric columns, and visualize them with a heatmap.
3. Build a student-grade DataFrame and compute average grade per subject using `groupby`.
4. Merge an orders DataFrame with a customers DataFrame and find customers with no orders.
5. Plot total sales by month using a bar chart from a time-series dataset.

---

## Review Questions

1. How do NumPy arrays differ from Python lists in storage and computation?
2. What does broadcasting allow, and when can it produce an unintended shape?
3. Why should missing-value and scaling decisions be based on the training data only?
4. When would you use `groupby`, a merge, or a pivot table in Pandas?
5. Which visualization would help reveal a distribution, a relationship, and a category comparison?

## Further Reading

- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
