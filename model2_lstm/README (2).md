# Model 2 — Bidirectional LSTM

## Overview

This model replaces the hand-engineered sentiment features of Assignment 1 with raw text
sequences processed by a **Bidirectional Long Short-Term Memory (BiLSTM)** network built
in Keras/TensorFlow. LSTMs are specifically designed to model sequential data with
long-range dependencies — exactly what is needed to understand nuanced sentiment in
multi-sentence movie reviews.

---

## File Structure

```
model2_lstm/
├── notebook_model2.ipynb      ← Main notebook (run this)
├── README.md                  ← This file
└── results/
    ├── results_model2.txt     ← Human-readable results
    ├── results_model2.json    ← Machine-readable results
    ├── best_lstm.keras        ← Saved best model weights
    └── model2_plots.png       ← Loss/accuracy/confusion-matrix plots
```

> The dataset should be placed at `../data/IMDB Dataset.csv`
> (one level up from this folder, inside a `data/` folder)

---

## Model Architecture

```
Input: integer token sequence (length = 250)
    │
    ▼
Embedding(vocab=20000, dim=128)       ← randomly initialized, trained end-to-end
    │
SpatialDropout1D(0.2)
    │
    ▼
Bidirectional LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
    │   forward LSTM(64) + backward LSTM(64) → concatenated 128-dim output
    ▼
Bidirectional LSTM(32, dropout=0.2, recurrent_dropout=0.2)
    │   forward LSTM(32) + backward LSTM(32) → concatenated 64-dim output
    ▼
Dense(64, activation='relu')
    │
Dropout(0.4)
    │
    ▼
Dense(1, activation='sigmoid')
```

**Total trainable parameters:** ~2.2 M

---

## Techniques Applied

### 1. Text Tokenization & Sequence Padding
Raw cleaned reviews are converted to integer sequences using Keras `Tokenizer`
(vocabulary capped at the top 20 000 words). Sequences are zero-padded or truncated to a
fixed length of 250 tokens.

### 2. Trainable Embedding Layer
A 128-dimensional embedding layer converts token indices to dense vectors, initialised
randomly and learned jointly with the rest of the network. This allows the model to
develop task-specific word representations without requiring external embeddings.

### 3. SpatialDropout1D
Drops entire feature channels rather than individual elements. This is more effective for
embeddings because it prevents correlated channels from co-adapting.

### 4. Bidirectional LSTM
Standard LSTMs only process sequences left-to-right. Bidirectional LSTMs run two
separate LSTMs in parallel — one forward, one backward — and concatenate their outputs.
This lets every token attend to both past and future context, which is important for
movie reviews where negations ("not bad") or qualifiers affect overall sentiment.

### 5. Stacked LSTM Layers
Two BiLSTM layers are stacked: the first returns full sequences (`return_sequences=True`)
so the second can process the temporal output, learning higher-level abstractions.

### 6. Dropout & Recurrent Dropout
Both standard dropout (applied to inputs/outputs) and recurrent dropout (applied to the
recurrent connections) are used at rate 0.2 to regularise the LSTM.

### 7. Adam Optimizer with ReduceLROnPlateau
Initial learning rate = 1e-3. The LR is halved when validation loss does not improve for
2 consecutive epochs, down to a minimum of 1e-6.

### 8. Early Stopping & Model Checkpointing
Training halts when validation loss stagnates for 3 epochs. The best model weights
(by validation accuracy) are saved to `results/best_lstm.keras`.

---

## Hyperparameters

| Hyperparameter        | Value      |
|-----------------------|------------|
| Vocabulary size       | 20 000     |
| Max sequence length   | 250        |
| Embedding dim         | 128        |
| LSTM units (L1 / L2)  | 64 / 32    |
| Dropout rate          | 0.2 – 0.4  |
| Batch size            | 128        |
| Max epochs            | 20         |
| Early stopping        | patience=3 |

---

## Results

See `results/results_model2.txt` for full metrics after running the notebook.

| Metric        | Assignment 1 Baseline | BiLSTM (Model 2) |
|---------------|-----------------------|------------------|
| Test Accuracy | ~0.71                 | see results      |
| F1-Score      | ~0.71                 | see results      |
| Precision     | ~0.70                 | see results      |
| Recall        | ~0.72                 | see results      |

---

## How to Run

```bash
pip install tensorflow pandas scikit-learn matplotlib seaborn beautifulsoup4
```

Open `notebook_model2.ipynb` in Jupyter and run all cells top to bottom.
Training takes approximately 15–25 minutes on CPU.
