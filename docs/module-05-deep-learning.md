---
title: "Module 5: Deep Learning"
sidebar_position: 5
---

# Module 5: Deep Learning

**Level:** Intermediate | **Prerequisites:** Module 4 (Machine Learning)

> Deep learning is a subfield of machine learning using neural networks with many layers to learn hierarchical representations from data. It powers image recognition, speech, translation, and generative AI.

---

## Learning Objectives

- Understand how artificial neural networks work
- Implement feedforward networks with TensorFlow/Keras
- Apply backpropagation and gradient descent conceptually
- Build CNNs for image classification
- Build RNNs/LSTMs for sequence modeling
- Prevent overfitting with regularization techniques

---

## 5.1 The Artificial Neuron

An artificial neuron computes a weighted sum of inputs, adds a bias, then passes through an **activation function**:

```
output = activation(w₁x₁ + w₂x₂ + ... + b)
```

### Activation Functions

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 200)

activations = {
    'Sigmoid': 1 / (1 + np.exp(-x)),
    'ReLU':    np.maximum(0, x),
    'Tanh':    np.tanh(x),
}

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax, (name, vals) in zip(axes, activations.items()):
    ax.plot(x, vals); ax.set_title(name); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
```

| Activation | Range | Use |
|---|---|---|
| Sigmoid | (0, 1) | Binary output layer |
| Softmax | (0, 1), sums to 1 | Multi-class output layer |
| ReLU | [0, ∞) | Hidden layers (most common) |
| Tanh | (-1, 1) | RNN hidden states |
| Leaky ReLU | (-∞, ∞) | Avoids dying ReLU problem |

---

## 5.2 Feedforward Neural Networks

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Build a simple MLP
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

model.summary()
```

### Training on MNIST

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize to [0, 1]
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test  = x_test.reshape(-1, 784).astype('float32') / 255.0

history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1,
)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

### Plotting Training Curves

```python
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Accuracy'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss'); plt.legend()
plt.show()
```

---

## 5.3 Backpropagation

Backpropagation computes the gradient of the loss with respect to every weight using the **chain rule**:

```
∂L/∂w = ∂L/∂output × ∂output/∂activation × ∂activation/∂w
```

Frameworks like TensorFlow and PyTorch compute these automatically (**autograd**). Understanding the concept helps you debug training issues.

**Common training issues:**
- **Vanishing gradients:** gradients shrink to zero in deep networks — use ReLU, batch norm, residual connections
- **Exploding gradients:** gradients grow too large — use gradient clipping
- **Dying ReLU:** neurons stuck at zero — use Leaky ReLU or ELU

---

## 5.4 Convolutional Neural Networks (CNNs)

CNNs are designed for spatial data (images). They use **convolution filters** to detect local patterns.

```python
model_cnn = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax'),
])

model_cnn.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Reshape for CNN: (samples, height, width, channels)
x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_test_cnn  = x_test.reshape(-1, 28, 28, 1)

model_cnn.fit(x_train_cnn, y_train, epochs=5, batch_size=64,
              validation_split=0.1)
```

### Transfer Learning

Use pre-trained models (ImageNet weights) as a starting point:

```python
base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
)
base_model.trainable = False   # freeze base layers

model_transfer = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax'),
])
```

---

## 5.5 Recurrent Neural Networks (RNNs)

RNNs process **sequential** data by maintaining a hidden state across time steps.

```python
# Sequence classification with LSTM
model_lstm = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=64),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dense(1, activation='sigmoid'),
])

model_lstm.compile(optimizer='adam', loss='binary_crossentropy',
                   metrics=['accuracy'])
```

**RNN variants:**
- **LSTM** (Long Short-Term Memory): handles long-range dependencies via gates
- **GRU** (Gated Recurrent Unit): simpler than LSTM, often comparable performance
- **Bidirectional LSTM**: processes sequence in both directions

---

## 5.6 Regularization Techniques

```python
# 1. Dropout — randomly zeros neurons during training
layers.Dropout(0.5)

# 2. L2 weight regularization
layers.Dense(64, activation='relu',
             kernel_regularizer=keras.regularizers.l2(0.01))

# 3. Batch Normalization — normalizes layer inputs
layers.BatchNormalization()

# 4. Early Stopping — stop when validation loss stops improving
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

# 5. Learning Rate Scheduling
lr_schedule = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3
)
```

---

## 5.7 Optimizers

| Optimizer | Notes |
|---|---|
| SGD | Simple, requires careful lr tuning |
| SGD + Momentum | Faster convergence than plain SGD |
| Adam | Adaptive lr per parameter — widely used |
| AdamW | Adam with weight decay — good default |
| RMSprop | Good for RNNs |

```python
# AdamW with weight decay
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
```

---

## Common Mistakes

- Not normalizing inputs before training
- Using sigmoid/tanh in deep hidden layers (vanishing gradients)
- Overfitting to small datasets — use more dropout or data augmentation
- Setting learning rate too high (training diverges) or too low (too slow)
- Forgetting to add `Flatten()` before `Dense()` after Conv layers

---

## Practice Exercises

1. Train the MNIST MLP model; experiment with different numbers of hidden layers and neurons.
2. Build a CNN for CIFAR-10 image classification; aim for > 75% test accuracy.
3. Implement early stopping and compare training curves with and without it.
4. Use transfer learning with MobileNetV2 to classify a small custom image dataset.
5. Build an LSTM for sentiment classification on the IMDB movie review dataset.

---

## Further Reading

- [Deep Learning (Goodfellow, Bengio, Courville — free online)](https://www.deeplearningbook.org/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [fast.ai Practical Deep Learning for Coders](https://course.fast.ai/)
- [Stanford CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
