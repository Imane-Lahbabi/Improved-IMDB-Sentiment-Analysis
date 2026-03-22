# Model 1 — Improved MLP with Optimization Techniques

## Overview

This model improves on the Assignment 1 baseline (simple 2-feature MLP) by applying
multiple optimization techniques while still training a pure NumPy multilayer perceptron.
The goal is to demonstrate that classical optimization methods can meaningfully close the
gap before moving to more powerful sequence models.

---

## File Structure

```
model1_improved_mlp/
├── notebook_model1.ipynb      ← Main notebook (run this)
├── README.md                  ← This file
└── results/
    ├── results_model1.txt     ← Human-readable results
    ├── results_model1.json    ← Machine-readable results
    └── model1_plots.png       ← Loss/accuracy/confusion-matrix plots
```

> The dataset should be placed at `../data/IMDB Dataset.csv`
> (one level up from this folder, inside a `data/` folder)

---

## Model Architecture

```
Input (5 features)
    │
    ▼
Dense(128) — He initialization
    │
BatchNorm → ReLU → Dropout(0.3)
    │
    ▼
Dense(64) — He initialization
    │
BatchNorm → ReLU → Dropout(0.3)
    │
    ▼
Dense(1) → Sigmoid
```

---

## Techniques Applied

### 1. Extended Feature Engineering (5 features instead of 2)
The original model used only VADER compound and TextBlob polarity. This model adds:
- **VADER compound score** — overall sentiment strength (−1 to +1)
- **TextBlob polarity** — lexicon-based polarity (−1 to +1)
- **TextBlob subjectivity** — degree of opinion vs. fact (0 to 1)
- **Exclamation mark density** — `count('!') / word_count`, captures emphatic writing
- **VADER pos-neg spread** — `pos_score − neg_score`, positive/negative balance

All features are standardised (zero mean, unit variance) using `sklearn.StandardScaler`
fitted only on the training split.

### 2. Deeper Architecture
Two hidden layers (128 → 64 neurons) replace the single 16-neuron hidden layer of the
baseline. Wider layers capture more feature interactions; the bottleneck design helps
learn a compressed sentiment representation.

### 3. He Weight Initialization
Weights are sampled from `N(0, sqrt(2/fan_in))` instead of `N(0, 0.01)`, which prevents
vanishing/exploding gradients at initialization when using ReLU activations.

### 4. Batch Normalization
Applied after each hidden linear transformation (before ReLU). Normalises activations to
have near-zero mean and unit variance per mini-batch, stabilising and accelerating training.

### 5. Dropout Regularization (rate = 0.3)
After each hidden layer's ReLU, 30% of neurons are randomly zeroed during training.
Prevents co-adaptation of neurons and reduces overfitting.

### 6. Adam Optimizer
Replaces vanilla SGD. Adam maintains per-parameter first (momentum) and second (RMS)
moment estimates, giving adaptive learning rates.
Hyperparameters: β₁ = 0.9, β₂ = 0.999, ε = 1e-8.

### 7. Cosine Learning Rate Scheduling
Learning rate follows a cosine annealing schedule:
`lr(t) = lr_min + 0.5 × (lr_base − lr_min) × (1 + cos(π × t / T))`
Starting LR = 3e-3, minimum LR = 1e-5.

### 8. Early Stopping
Training halts when validation loss does not improve for 15 consecutive epochs.
Best weights are restored at the end of training.

### 9. Mini-Batch Training (batch size = 64)
Training data is shuffled and processed in mini-batches of 64 samples per update.

---

## Hyperparameters

| Hyperparameter     | Value        |
|--------------------|--------------|
| Hidden layers      | 2            |
| Layer widths       | 128 → 64     |
| Dropout rate       | 0.3          |
| Base learning rate | 3e-3         |
| Batch size         | 64           |
| Max epochs         | 150          |
| Early stopping     | patience=15  |

---

## Results

See `results/results_model1.txt` for full metrics after running the notebook.

| Metric        | Assignment 1 Baseline | Model 1 (Improved MLP) |
|---------------|-----------------------|------------------------|
| Test Accuracy | ~0.71                 | see results            |
| F1-Score      | ~0.71                 | see results            |
| Precision     | ~0.70                 | see results            |
| Recall        | ~0.72                 | see results            |

---

## How to Run

```bash
pip install numpy pandas scikit-learn matplotlib seaborn vaderSentiment textblob beautifulsoup4
```

Open `notebook_model1.ipynb` in Jupyter and run all cells top to bottom.
