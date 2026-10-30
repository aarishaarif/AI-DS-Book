---
title: "Module 8: Generative AI and Large Language Models"
sidebar_position: 8
---

# Module 8: Generative AI and Large Language Models

**Level:** Intermediate–Advanced | **Prerequisites:** Module 5 (Deep Learning)

> Generative AI produces new content — text, images, code, audio — by learning patterns from vast training data. Large Language Models (LLMs) are the transformer-based foundation of modern AI assistants.

---

## Learning Objectives

- Understand the Transformer architecture and attention mechanism
- Use LLMs via APIs (Gemini, GPT-4) for text generation
- Apply prompt engineering techniques effectively
- Build a Retrieval-Augmented Generation (RAG) pipeline
- Fine-tune LLMs for specific tasks
- Evaluate and safeguard LLM outputs

## How Generative AI Fits Together

An LLM receives tokens, transforms their representations through attention layers, and predicts a continuation. Embeddings represent meaning as vectors so related text can be found. Prompting shapes the task at runtime, fine-tuning changes model behavior through training, and RAG supplies selected context at answer time.

| Term | Meaning |
| --- | --- |
| Token | A unit of text processed by a language model |
| Embedding | A vector representation used to compare meaning |
| Attention | A mechanism that weighs relationships between tokens |
| Context window | The input and output space available for a request |
| Grounding | Tying an answer to evidence from a trusted source |

RAG follows a clear sequence: chunk source documents, embed and store chunks, retrieve candidates, check relevance, and generate only from accepted evidence.

---

## 8.1 The Transformer Architecture

Introduced in *"Attention Is All You Need"* (Vaswani et al., 2017), the Transformer is the foundation of all modern LLMs.

### Key Components

```
Input Tokens
    → Embeddings + Positional Encoding
        → Multi-Head Self-Attention
            → Feed-Forward Network
                → Layer Normalization
                    → Output
```

### Self-Attention (Conceptual)

Self-attention allows each token to "look at" every other token when building its representation:

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Q: Query matrix
    K: Key matrix
    V: Value matrix
    d_k: dimension of keys
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)           # Scale
    weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)  # Softmax
    return weights @ V                         # Weighted sum of values

# Example with 3 tokens, dimension 4
Q = np.random.randn(3, 4)
K = np.random.randn(3, 4)
V = np.random.randn(3, 4)

output = scaled_dot_product_attention(Q, K, V)
print("Output shape:", output.shape)  # (3, 4)
```

**Why attention is powerful:** Each output token is a weighted combination of all input tokens, allowing the model to relate distant words directly — unlike RNNs which process sequentially.

---

## 8.2 Working with LLMs via API

### Google Gemini API

```python
import google.generativeai as genai
import os

genai.configure(api_key=os.environ['GEMINI_API_KEY'])

model = genai.GenerativeModel('gemini-2.5-flash')
response = model.generate_content("Explain gradient descent in simple terms.")
print(response.text)
```

### Chat (Multi-turn Conversation)

```python
chat = model.start_chat(history=[])

response1 = chat.send_message("I'm learning about neural networks. Can you start with the basics?")
print(response1.text)

response2 = chat.send_message("Now explain backpropagation.")
print(response2.text)

# The model remembers the conversation context
```

### Controlling Output

```python
import google.generativeai as genai
from google.generativeai import types

model = genai.GenerativeModel(
    'gemini-2.5-flash',
    generation_config=types.GenerationConfig(
        temperature=0.2,         # Lower = more deterministic
        max_output_tokens=512,
        top_p=0.95,
    ),
    system_instruction="You are a helpful assistant that explains technical concepts clearly.",
)

response = model.generate_content("What is a convolutional layer?")
print(response.text)
```

---

## 8.3 Embeddings

Embeddings convert text into dense numerical vectors, capturing semantic meaning.

```python
import google.generativeai as genai
import numpy as np

genai.configure(api_key=os.environ['GEMINI_API_KEY'])

def get_embedding(text: str) -> list:
    result = genai.embed_content(
        model='models/gemini-embedding-001',
        content=text,
        task_type='retrieval_document',
    )
    return result['embedding']

# Compare semantic similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

e1 = get_embedding("What is machine learning?")
e2 = get_embedding("Explain ML algorithms")
e3 = get_embedding("How do I cook pasta?")

print(f"ML vs ML:    {cosine_similarity(e1, e2):.4f}")  # High
print(f"ML vs Pasta: {cosine_similarity(e1, e3):.4f}")  # Low
```

---

## 8.4 Prompt Engineering

Prompt engineering is the practice of crafting inputs to get better outputs from LLMs.

### Zero-Shot Prompting

```python
response = model.generate_content(
    "Classify the sentiment of this review as positive, negative, or neutral:\n"
    "The course was incredibly well structured and easy to follow."
)
# Expected: Positive
```

### Few-Shot Prompting

```python
few_shot_prompt = """
Classify sentiment. Examples:

Text: "This product is amazing!"
Sentiment: Positive

Text: "Worst experience ever."
Sentiment: Negative

Text: "It was okay, nothing special."
Sentiment: Neutral

Text: "The delivery was super fast and the item works perfectly."
Sentiment:"""

response = model.generate_content(few_shot_prompt)
```

### Chain-of-Thought Prompting

```python
cot_prompt = """
Solve this step by step:

A train travels 120 km in 2 hours. A car travels the same distance in 1.5 hours. 
How much faster is the car than the train in km/h?

Let's think step by step:"""

response = model.generate_content(cot_prompt)
```

### System Instructions and Role Prompting

```python
model = genai.GenerativeModel(
    'gemini-2.5-flash',
    system_instruction="""
You are an expert Python tutor. Your responses should:
- Be concise and beginner-friendly
- Include runnable code examples
- Point out common mistakes
- Never provide just theory without examples
"""
)
```

---

## 8.5 Retrieval-Augmented Generation (RAG)

RAG improves LLM accuracy by grounding answers in retrieved documents rather than relying on parametric memory.

:::info Why the pipeline matters
The question is converted into an embedding so the system can compare meaning rather than only exact words. The closest chunks become context for the model; the model is instructed to answer from that context and return source sections.
:::

```
User Question
    → Embed question
        → Search vector database for similar chunks
            → Retrieve top-k relevant chunks
                → Build prompt: "Answer based only on this context: {chunks}"
                    → Generate grounded answer
                        → Return answer + citations
```

### Building a Simple RAG System

```python
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
import os

genai.configure(api_key=os.environ['GEMINI_API_KEY'])

# 1. Create vector store
client = QdrantClient(":memory:")  # in-memory for demo
client.create_collection(
    collection_name="docs",
    vectors_config=qdrant_models.VectorParams(size=768, distance=qdrant_models.Distance.COSINE),
)

# 2. Index documents
docs = [
    "Python is a high-level programming language known for its simplicity.",
    "NumPy provides efficient numerical arrays for scientific computing.",
    "Pandas DataFrames enable flexible tabular data manipulation.",
    "Machine learning uses algorithms to learn patterns from data.",
]

for i, doc in enumerate(docs):
    embedding = genai.embed_content(
        model='models/gemini-embedding-001',
        content=doc,
        task_type='retrieval_document',
    )['embedding']

    client.upsert(collection_name="docs", points=[
        qdrant_models.PointStruct(id=i, vector=embedding, payload={"text": doc})
    ])

# 3. Answer a question
def answer_with_rag(question: str) -> str:
    # Embed question
    q_embedding = genai.embed_content(
        model='models/gemini-embedding-001',
        content=question,
        task_type='retrieval_query',
    )['embedding']

    # Retrieve relevant chunks
    results = client.search(collection_name="docs", query_vector=q_embedding, limit=2)
    context = "\n\n".join([r.payload["text"] for r in results])

    # Generate grounded answer
    gen_model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"""Answer the question using ONLY the context below.
Context:
{context}

Question: {question}
Answer:"""
    response = gen_model.generate_content(prompt)
    return response.text

print(answer_with_rag("What is Pandas used for?"))
```

:::tip How this code works
The collection stores one vector per document. At question time, the question gets its own vector, Qdrant finds nearby vectors using cosine similarity, and only those retrieved documents are passed to the generator. This makes retrieval inspectable and lets the answer include citations.
:::

---

## 8.6 Fine-Tuning LLMs

Fine-tuning adapts a pre-trained model to a specific domain or task by training on labeled examples.

### When to Fine-Tune vs. Prompt Engineer

| Scenario | Recommended Approach |
|---|---|
| Task needs specific style or format | Fine-tuning |
| Domain knowledge is rare in training data | Fine-tuning |
| Few examples (< 20) | Few-shot prompting |
| Quick prototyping | Prompt engineering |
| Privacy-sensitive data | Fine-tune local model |

### Parameter-Efficient Fine-Tuning (LoRA)

LoRA (Low-Rank Adaptation) fine-tunes a small number of additional parameters instead of updating all weights:

```python
# Conceptual — using HuggingFace + PEFT library
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

lora_config = LoraConfig(
    r=16,              # rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
# trainable params: 8,388,608 (0.48% of total)
```

---

## 8.7 Evaluating LLM Outputs

LLM evaluation is harder than traditional ML — outputs are often open-ended.

```python
# Automated evaluation: ROUGE score for summarization
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

reference = "Python is widely used in data science and machine learning."
generated = "Python is a popular language for machine learning and data analysis."

scores = scorer.score(reference, generated)
print(scores)
```

### LLM-as-Judge

```python
def evaluate_with_llm(question, answer, context):
    judge_prompt = f"""
Rate this answer on a scale of 1-5 for:
1. Factual accuracy (grounded in context)
2. Completeness
3. Clarity

Context: {context}
Question: {question}
Answer: {answer}

Provide a JSON response: {{"accuracy": X, "completeness": X, "clarity": X, "reasoning": "..."}}
"""
    response = model.generate_content(judge_prompt)
    return response.text
```

---

## Common Mistakes

- Using overly long, vague prompts — be specific and concise
- Not setting `temperature=0` for factual/deterministic tasks
- Trusting LLM output without grounding (hallucination risk)
- Fine-tuning when prompt engineering would suffice
- Exposing the system prompt or API key in client-side code

---

## Practice Exercises

1. Use the Gemini API to build a text summarizer that takes an article and returns 3 key points.
2. Implement few-shot classification for 5 categories of customer feedback.
3. Build the RAG system above with your own documents (e.g., the content of Module 1).
4. Compare the cosine similarity of 10 sentence pairs and visualize as a heatmap.
5. Write a prompt that extracts structured JSON data (name, date, amount) from unstructured invoice text.

---

## Review Questions

1. How does self-attention let a transformer use context from other tokens?
2. What does an embedding represent, and why compare embeddings with cosine similarity?
3. How does RAG combine retrieval with language generation?
4. When might fine-tuning be preferable to retrieval or prompt changes?
5. Why should generated output be evaluated for factuality as well as fluency?

## Further Reading

- [Attention Is All You Need (paper)](https://arxiv.org/abs/1706.03762)
- [Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [RLHF — Reinforcement Learning from Human Feedback (Anthropic)](https://arxiv.org/abs/2204.05862)
