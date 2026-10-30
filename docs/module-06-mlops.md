---
title: "Module 6: MLOps"
sidebar_position: 6
---

# Module 6: MLOps

**Level:** Intermediate | **Prerequisites:** Module 5 (Deep Learning)

> MLOps (Machine Learning Operations) is the practice of deploying, monitoring, and maintaining ML models in production reliably. It combines software engineering, DevOps, and data science.

---

## Learning Objectives

- Understand the gap between a notebook model and a production system
- Structure ML projects for reproducibility and maintainability
- Track experiments and model versions with MLflow
- Serve models with FastAPI and Docker
- Monitor data drift and model degradation in production
- Apply CI/CD principles to ML pipelines

## Why MLOps Exists

Training a model is one step in a longer system. Data arrives, code changes, models are registered, an API serves predictions, and monitoring detects when reality no longer matches training assumptions. MLOps makes these steps repeatable so a useful experiment can become a dependable service.

| Term | Meaning |
| --- | --- |
| Reproducibility | Recreating a result from recorded inputs and versions |
| Artifact | A stored output such as a model or evaluation report |
| Deployment | Making a model available to a consumer |
| Data drift | A change in the distribution of input data |
| Concept drift | A change in the relationship between inputs and target |

---

## 6.1 The MLOps Problem

A model that works in a notebook often fails in production because:

| Problem | Cause |
|---|---|
| Not reproducible | Random seeds, environment differences |
| Hard to retrain | No automated pipeline |
| Silent failures | No monitoring for input drift |
| Slow iteration | Manual deployment process |
| No rollback | No versioning |

MLOps solves this with structure, automation, and monitoring.

---

## 6.2 Project Structure

A well-structured ML project:

```
ml_project/
├── data/
│   ├── raw/            # Original, immutable data
│   ├── processed/      # Cleaned, feature-engineered
│   └── external/       # Third-party data
├── models/             # Saved model artifacts
├── notebooks/          # Exploratory analysis
├── src/
│   ├── data.py         # Data loading and preprocessing
│   ├── features.py     # Feature engineering
│   ├── train.py        # Training pipeline
│   ├── evaluate.py     # Evaluation and metrics
│   └── predict.py      # Inference
├── tests/              # Unit and integration tests
├── configs/            # YAML configuration files
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 6.3 Experiment Tracking with MLflow

MLflow tracks parameters, metrics, artifacts, and code for every experiment run.

```bash
pip install mlflow
```

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_experiment("iris-classification")

with mlflow.start_run(run_name="random_forest_v1"):
    # Load data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Parameters
    n_estimators = 100
    max_depth = 5

    # Train
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    accuracy = accuracy_score(y_test, model.predict(X_test))

    # Log to MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

    print(f"Accuracy: {accuracy:.4f}")
```

```bash
# Launch the MLflow UI to compare runs
mlflow ui
# Open http://localhost:5000
```

---

## 6.4 Model Serialization

### Saving and Loading with joblib

```python
import joblib
from sklearn.ensemble import RandomForestClassifier

# Save
model = RandomForestClassifier()
model.fit(X_train, y_train)
joblib.dump(model, 'models/rf_model.joblib')

# Load
loaded_model = joblib.load('models/rf_model.joblib')
predictions = loaded_model.predict(X_test)
```

### Saving with pickle (standard library)

```python
import pickle

with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
```

### ONNX (cross-framework portability)

```bash
pip install skl2onnx onnxruntime
```

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt

# Convert to ONNX
initial_type = [('float_input', FloatTensorType([None, 4]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

with open("models/model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# Run inference with ONNX Runtime
sess = rt.InferenceSession("models/model.onnx")
pred = sess.run(None, {"float_input": X_test.astype('float32')})
```

---

## 6.5 Model Serving with FastAPI

```python
# serve.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Iris Classifier API")
model = joblib.load("models/rf_model.joblib")

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class Prediction(BaseModel):
    class_id: int
    class_name: str
    confidence: float

CLASS_NAMES = ['setosa', 'versicolor', 'virginica']

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=Prediction)
def predict(features: IrisFeatures):
    X = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]])
    class_id = int(model.predict(X)[0])
    probas = model.predict_proba(X)[0]
    return Prediction(
        class_id=class_id,
        class_name=CLASS_NAMES[class_id],
        confidence=float(probas[class_id]),
    )
```

```bash
uvicorn serve:app --host 0.0.0.0 --port 8000 --reload
```

---

## 6.6 Containerization with Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ models/
COPY serve.py .

EXPOSE 8000
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t iris-classifier .
docker run -p 8000:8000 iris-classifier
```

---

## 6.7 Data Validation with Pydantic

Always validate inputs at the API boundary:

```python
from pydantic import BaseModel, Field, validator

class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, lt=20, description="cm")
    sepal_width:  float = Field(..., gt=0, lt=20)
    petal_length: float = Field(..., gt=0, lt=20)
    petal_width:  float = Field(..., gt=0, lt=20)

    @validator('sepal_length', 'sepal_width', 'petal_length', 'petal_width')
    def must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Feature must be positive')
        return v
```

---

## 6.8 Monitoring and Drift Detection

After deployment, models degrade because the real world changes.

```python
# Concept drift: relationship between features and labels changes
# Data drift: input distribution shifts

from scipy.stats import ks_2samp

def detect_data_drift(reference: list, production: list, threshold=0.05):
    stat, p_value = ks_2samp(reference, production)
    drifted = p_value < threshold
    return {'statistic': stat, 'p_value': p_value, 'drift_detected': drifted}

result = detect_data_drift(X_train[:, 0], X_test[:, 0])
print(result)
```

**Production monitoring checklist:**
- Log all predictions and inputs
- Track accuracy/precision/recall over time
- Alert on significant distribution shifts (Kolmogorov-Smirnov, PSI)
- Schedule periodic retraining
- Implement A/B testing for model updates

---

## 6.9 CI/CD for ML

```yaml
# .github/workflows/ml-ci.yml
name: ML CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/ -v
      - name: Train model
        run: python src/train.py
      - name: Evaluate model
        run: python src/evaluate.py --threshold 0.90
```

---

## Common Mistakes

- Shipping a notebook as "production code"
- Not versioning models — can't roll back after a bad deployment
- Skipping input validation — malformed inputs crash the API
- No monitoring — model degrades silently for weeks
- Fitting preprocessing on test data (data leakage)

---

## Practice Exercises

1. Log 3 different hyperparameter experiments with MLflow and compare them in the UI.
2. Build a FastAPI endpoint that serves predictions from a trained scikit-learn model.
3. Dockerize the FastAPI model server and confirm it responds to `POST /predict`.
4. Implement a data drift detector using Kolmogorov-Smirnov test on a sample dataset.
5. Write a GitHub Actions workflow that trains a model, evaluates it, and fails if accuracy < 0.85.

---

## Review Questions

1. Why record data, code, parameters, and metrics for each experiment?
2. What changes when a model moves from a notebook into a prediction service?
3. How do containers help make a model service reproducible?
4. What can data drift monitoring detect, and what can it not prove?
5. Why should a deployment have versioning and a rollback plan?

## Further Reading

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Made With ML (MLOps Course)](https://madewithml.com/)
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/)
- [Evidently AI — Data Drift Monitoring](https://docs.evidentlyai.com/)
