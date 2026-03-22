# Model 3 — 1D CNN with Pretrained Word2Vec Embeddings

## Overview

This model trains a **TextCNN** — a 1D Convolutional Neural Network with parallel
multi-kernel convolutions — initialised with **pretrained Word2Vec embeddings** generated
using the Gensim library. CNNs excel at detecting local n-gram patterns regardless of
their position in the sequence, making them highly effective for text classification
while being significantly faster to train than RNNs.

---

## File Structure

```
model3_1dcnn_word2vec/
├── notebook_model3.ipynb      ← Main notebook (run this)
├── README.md                  ← This file
└── results/
    ├── results_model3.txt     ← Human-readable results
    ├── results_model3.json    ← Machine-readable results
    ├── best_cnn.keras         ← Saved best model weights
    ├── word2vec_imdb.model    ← Trained Word2Vec model
    └── model3_plots.png       ← Loss/accuracy/confusion-matrix plots
```

> The dataset should be placed at `../data/IMDB Dataset.csv`
> (one level up from this folder, inside a `data/` folder)

---

## Model Architecture

```
Input: integer token sequence (length = 250)
    │
    ▼
Embedding(vocab=20000, dim=300, weights=Word2Vec, trainable=True)
    │
    ├─────────────────────┬─────────────────────┐
    ▼                     ▼                     ▼
Conv1D(128, k=2)    Conv1D(128, k=3)    Conv1D(128, k=4)
BatchNorm + ReLU    BatchNorm + ReLU    BatchNorm + ReLU
GlobalMaxPool1D     GlobalMaxPool1D     GlobalMaxPool1D
    │                     │                     │
    └─────────────────────┴─────────────────────┘
                          ▼
                    Concatenate → (384,)
                          │
                    Dropout(0.5)
                          │
                    Dense(128, relu)
                          │
                    BatchNormalization
                          │
                    Dropout(0.25)
                          │
                    Dense(1, sigmoid)
```

**Total trainable parameters:** ~7.5 M (dominated by the 300-dim embedding matrix)

---

## Techniques Applied

### 1. Gensim Word2Vec Embeddings
A Skip-gram Word2Vec model is trained on the full IMDB corpus before model training.
Key settings:
- `vector_size = 300` — high-dimensional representation captures subtle word semantics
- `window = 5` — each word learns from ±5 surrounding words
- `min_count = 2` — filters words appearing only once
- `sg = 1` — Skip-gram; better for rare words than CBOW
- `epochs = 10` — multiple passes over the corpus

The resulting 300-dimensional word vectors encode semantic similarity so that synonyms
are close together in the embedding space. Initialising the embedding layer with these
vectors gives the CNN a head-start compared to random initialisation.

### 2. Fine-Tunable Embedding Layer (`trainable=True`)
Although the embedding matrix is seeded with Word2Vec vectors, gradients are allowed to
flow through it during training. This lets the network adapt the general-purpose word
representations to the specific sentiment task.

### 3. Multi-Kernel TextCNN (Parallel Convolutions)
Three parallel 1D convolutional branches use kernel sizes 2, 3, and 4, capturing:
- **Bigrams (k=2):** short phrase patterns ("not good", "great acting")
- **Trigrams (k=3):** three-word phrases ("really quite bad", "one of best")
- **4-grams (k=4):** longer local patterns and negation scopes

Each branch uses 128 filters, and their outputs are concatenated into a 384-dim vector.

### 4. Global Max Pooling
After each conv branch, `GlobalMaxPool1D` selects the maximum activation across all
positions, capturing the most salient n-gram feature regardless of where it appears
in the review.

### 5. Batch Normalization
Applied after each convolution and after the dense layer to stabilise training.

### 6. Dropout (0.5 after concat, 0.25 after dense)
Heavy dropout on the concatenated feature vector prevents filters from co-adapting
across different kernel sizes.

### 7. Adam Optimizer with ReduceLROnPlateau
Initial LR = 1e-3. LR is halved when val_loss plateaus for 2 consecutive epochs,
minimum LR = 1e-6.

### 8. Early Stopping & Model Checkpointing
Training halts after 4 epochs of no validation loss improvement. Best weights are
saved to `results/best_cnn.keras`.

---

## Hyperparameters

| Hyperparameter        | Value          |
|-----------------------|----------------|
| Vocabulary size       | 20 000         |
| Max sequence length   | 250            |
| Embedding dim         | 300 (Word2Vec) |
| Conv filters          | 128 per branch |
| Kernel sizes          | 2, 3, 4        |
| Dropout rate          | 0.5 / 0.25     |
| Batch size            | 256            |
| Max epochs            | 20             |
| Early stopping        | patience=4     |

---

## Results

See `results/results_model3.txt` for full metrics after running the notebook.

| Metric        | Assignment 1 Baseline | 1D CNN + W2V (Model 3) |
|---------------|-----------------------|------------------------|
| Test Accuracy | ~0.71                 | see results            |
| F1-Score      | ~0.71                 | see results            |
| Precision     | ~0.70                 | see results            |
| Recall        | ~0.72                 | see results            |

---

## How to Run

```bash
pip install tensorflow gensim pandas scikit-learn matplotlib seaborn beautifulsoup4
```

Open `notebook_model3.ipynb` in Jupyter and run all cells top to bottom.
Word2Vec training runs in ~2–5 minutes. CNN training takes ~5–10 minutes on CPU.

---

## References

- Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. *EMNLP 2014*.
- Mikolov, T. et al. (2013). Efficient Estimation of Word Representations in Vector Space. *ICLR 2013*.
