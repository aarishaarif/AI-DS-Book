import React, { useState } from 'react';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import CodeBlock from '@theme/CodeBlock';
import './projects.css';

interface Project {
  id: string;
  title: string;
  module: string;
  moduleUrl: string;
  level: 'Beginner' | 'Intermediate' | 'Advanced';
  time: string;
  summary: string;
  skills: string[];
  steps: string[];
  starterCode: string;
}

const PROJECTS: Project[] = [
  {
    id: 'eda-data-cleaning',
    title: 'Automated Exploratory Data Analysis & Cleaning Pipeline',
    module: 'Module 2: Python for Data Science',
    moduleUrl: '/docs/module-02-python-for-data-science',
    level: 'Beginner',
    time: '2–3 hours',
    summary:
      'Build an end-to-end data ingestion, cleaning, and statistical profile generator using NumPy and Pandas. Detect missing values, impute outliers, and generate automated visual reports.',
    skills: ['Python', 'NumPy', 'Pandas', 'Data Cleaning', 'Matplotlib'],
    steps: [
      'Ingest raw messy CSV records with inconsistent date formats and missing numeric fields.',
      'Compute descriptive statistics and detect distribution skewness.',
      'Handle missing values with median imputation and drop redundant categorical columns.',
      'Export a clean, normalized dataset ready for downstream machine learning tasks.',
    ],
    starterCode: `import pandas as pd
import numpy as np

def clean_dataset(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    # 1. Fill missing numeric values with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    # 2. Strip whitespace from string columns
    str_cols = df.select_dtypes(include=['object']).columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())
    
    return df`,
  },
  {
    id: 'customer-churn-ml',
    title: 'Customer Churn Prediction with Scikit-Learn',
    module: 'Module 4: Machine Learning',
    moduleUrl: '/docs/module-04-machine-learning',
    level: 'Intermediate',
    time: '3–4 hours',
    summary:
      'Develop an end-to-end supervised machine learning pipeline to classify customer attrition risk. Perform feature scaling, train Random Forest and Logistic Regression models, and evaluate ROC-AUC.',
    skills: ['Scikit-Learn', 'Classification', 'Feature Scaling', 'ROC-AUC', 'Cross-Validation'],
    steps: [
      'Split tabular customer records into training and test sets using stratified sampling.',
      'Construct a scikit-learn Pipeline with StandardScaler and OneHotEncoder.',
      'Train a RandomForestClassifier with hyperparameter tuning via GridSearchCV.',
      'Analyze feature importances and compute confusion matrix metrics.',
    ],
    starterCode: `from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# X: feature matrix, y: churn binary label (0 or 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))`,
  },
  {
    id: 'deep-learning-vision',
    title: 'Convolutional Neural Network for Image Recognition',
    module: 'Module 5: Deep Learning',
    moduleUrl: '/docs/module-05-deep-learning',
    level: 'Intermediate',
    time: '4–5 hours',
    summary:
      'Construct a convolutional neural network (CNN) from scratch using PyTorch. Understand convolutional kernels, pooling layers, forward pass, backpropagation, and loss convergence.',
    skills: ['PyTorch', 'CNN', 'Backpropagation', 'Computer Vision', 'Optimizers'],
    steps: [
      'Load image data and apply transformations (normalization, data augmentation).',
      'Define Conv2d layers, ReLU activations, and MaxPool2d downsampling.',
      'Set up CrossEntropyLoss and the Adam optimizer with weight decay.',
      'Write the training loop tracking validation loss across epochs.',
    ],
    starterCode: `import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))`,
  },
  {
    id: 'mlops-pipeline',
    title: 'Reproducible MLOps Pipeline with MLflow & Docker',
    module: 'Module 6: MLOps',
    moduleUrl: '/docs/module-06-mlops',
    level: 'Advanced',
    time: '5–6 hours',
    summary:
      'Package a model into an enterprise-ready continuous pipeline. Track hyperparameters, artifacts, and metrics using MLflow, package the inference API into Docker, and automate health probes.',
    skills: ['MLflow', 'Docker', 'FastAPI', 'Model Serving', 'CI/CD'],
    steps: [
      'Instrument training runs with MLflow to log parameters, loss curves, and artifact weights.',
      'Register the champion model to the MLflow Model Registry.',
      'Wrap model inference inside a high-throughput FastAPI microservice.',
      'Containerize the API with a multi-stage Dockerfile and healthcheck probes.',
    ],
    starterCode: `import mlflow
import mlflow.sklearn

with mlflow.start_run():
    mlflow.log_param("n_estimators", 150)
    mlflow.log_param("learning_rate", 0.05)
    
    # Train model
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    
    mlflow.log_metric("accuracy", score)
    mlflow.sklearn.log_model(model, "churn-model")`,
  },
  {
    id: 'grounded-rag-system',
    title: 'Production RAG System with Vector Search & Citations',
    module: 'Module 8: Generative AI and LLMs',
    moduleUrl: '/docs/module-08-generative-ai-and-llms',
    level: 'Advanced',
    time: '4–5 hours',
    summary:
      'Build a strictly grounded Retrieval-Augmented Generation (RAG) system using Google Gemini Embeddings, Qdrant vector database, and prompt engineering with verifiable source citations.',
    skills: ['RAG', 'Vector Search', 'Qdrant', 'Gemini API', 'Prompt Engineering'],
    steps: [
      'Chunk educational markdown documents preserving heading and metadata hierarchies.',
      'Generate dense vector embeddings using Google Gemini embeddings API.',
      'Index points into Qdrant collection with cosine distance metric.',
      'Implement evidence scoring threshold to reject out-of-domain questions.',
    ],
    starterCode: `from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# Vector retrieval with similarity threshold
results = qdrant.query_points(
    collection_name="ai_ds_learning_book",
    query=query_vector,
    limit=5,
    score_threshold=0.70  # Strict evidence requirement
)

chunks = [pt.payload["content"] for pt in results.points]
# Pass to generation prompt with verified citations`,
  },
  {
    id: 'agentic-ai-assistant',
    title: 'Autonomous Tool-Calling Agent with ReAct Loops',
    module: 'Module 9: Agentic AI',
    moduleUrl: '/docs/module-09-agentic-ai',
    level: 'Advanced',
    time: '5–6 hours',
    summary:
      'Create an autonomous multi-step agent using the ReAct (Reason + Act) loop. Equip the agent with Python code execution, calculator tools, and document lookup tools to solve complex queries.',
    skills: ['Agentic AI', 'ReAct Loop', 'Function Calling', 'Tool Use', 'Multi-Agent'],
    steps: [
      'Define function call schemas for calculation, web lookup, and database querying.',
      'Implement Thought -> Action -> Observation loop.',
      'Handle error recovery when tool executions return invalid responses.',
      'Implement termination criteria and final grounded answer synthesis.',
    ],
    starterCode: `class ReActAgent:
    def __init__(self, tools, llm):
        self.tools = tools
        self.llm = llm

    def run(self, task: str, max_iterations: int = 5):
        history = [f"Task: {task}"]
        for _ in range(max_iterations):
            thought_action = self.llm.decide(history)
            if thought_action.is_final:
                return thought_action.answer
            
            observation = self.tools[thought_action.tool_name](*thought_action.args)
            history.append(f"Action: {thought_action.tool_name} -> {observation}")
        return "Max iterations reached."`,
  },
];

export default function ProjectsPage(): JSX.Element {
  const [selectedLevel, setSelectedLevel] = useState<string>('All');

  const filteredProjects =
    selectedLevel === 'All'
      ? PROJECTS
      : PROJECTS.filter((p) => p.level === selectedLevel);

  return (
    <Layout
      title="Guided AI/ML Projects | AI & Data Science Learning Book"
      description="Practical, portfolio-ready guided machine learning and AI projects with starter code and architectures."
    >
      <main className="projects-container">
        <header className="projects-header">
          <span className="badge-tag">Hands-on Experience</span>
          <Heading as="h1">Guided AI & Data Science Projects</Heading>
          <p className="projects-subtitle">
            Solidify your knowledge by building portfolio-worthy projects. Each project is tied to specific curriculum modules and includes clear implementation milestones and starter code.
          </p>

          <div className="filter-bar">
            {['All', 'Beginner', 'Intermediate', 'Advanced'].map((lvl) => (
              <button
                key={lvl}
                className={`filter-btn ${selectedLevel === lvl ? 'active' : ''}`}
                onClick={() => setSelectedLevel(lvl)}
              >
                {lvl}
              </button>
            ))}
          </div>
        </header>

        <div className="projects-grid">
          {filteredProjects.map((p) => (
            <article key={p.id} className="project-card">
              <div className="card-top">
                <span className={`level-badge level-${p.level.toLowerCase()}`}>
                  {p.level}
                </span>
                <span className="time-badge">⏱ {p.time}</span>
              </div>

              <Heading as="h2" className="project-title">
                {p.title}
              </Heading>

              <div className="module-link-wrapper">
                📚 Aligned with:{' '}
                <a href={p.moduleUrl} className="module-link">
                  {p.module}
                </a>
              </div>

              <p className="project-summary">{p.summary}</p>

              <div className="skills-tags">
                {p.skills.map((skill, sIdx) => (
                  <span key={sIdx} className="skill-pill">
                    {skill}
                  </span>
                ))}
              </div>

              <div className="steps-block">
                <h4>Implementation Milestones</h4>
                <ol className="steps-list">
                  {p.steps.map((st, stIdx) => (
                    <li key={stIdx}>{st}</li>
                  ))}
                </ol>
              </div>

              <div className="code-block-wrapper">
                <h4>Starter Architecture</h4>
                <CodeBlock language="python">{p.starterCode}</CodeBlock>
              </div>

              <div className="card-actions">
                <a href={p.moduleUrl} className="btn-read-module">
                  Review Lesson Material →
                </a>
                <a href="/chat" className="btn-ask-ai">
                  Ask RAG Chatbot About This →
                </a>
              </div>
            </article>
          ))}
        </div>
      </main>
    </Layout>
  );
}
