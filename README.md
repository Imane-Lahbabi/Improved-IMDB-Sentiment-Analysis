# Assignment 2 — IMDB Sentiment Analysis: Model Comparison

## Repository Structure

```
Assignment 2/
├── README.md
├── data/
│   └── IMDB Dataset.csv
│
├── model1_improved_mlp/
│   ├── notebook_model1.ipynb
│   ├── README.md
│   └── results/
│       ├── results_model1.txt
│       ├── results_model1.json
│       └── model1_plots.png
│
├── model2_lstm/
│   ├── notebook_model2.ipynb
│   ├── README.md
│   └── results/
│       ├── results_model2.txt
│       ├── results_model2.json
│       ├── best_lstm.keras
│       └── model2_plots.png
│
└── model3_1dcnn_word2vec/
    ├── notebook_model3.ipynb
    ├── README.md
    └── results/
        ├── results_model3.txt
        ├── results_model3.json
        ├── best_cnn.keras
        ├── word2vec_imdb.model
        └── model3_plots.png
```

---

## Models at a Glance

| # | Model                   | Key Technique(s)                                    | Test Accuracy |
|---|-------------------------|-----------------------------------------------------|---------------|
| 1 | Improved MLP            | Adam, Dropout, BatchNorm, He init, 5 features       | see results   |
| 2 | Bidirectional LSTM      | Sequence model, BiLSTM layers, trainable embedding  | see results   |
| 3 | 1D CNN + Word2Vec       | Multi-kernel TextCNN, pretrained W2V embeddings     | see results   |

---

## Dataset

**IMDB Movie Review Sentiment Dataset** — 50 000 reviews (25k positive, 25k negative).
After deduplication: ~49 582 reviews.
Place the CSV at: `data/IMDB Dataset.csv`

Splits used consistently across all models:
- Train: 60% (~29 700 reviews)
- Validation: 20% (~9 900 reviews)
- Test: 20% (~9 900 reviews)
(Stratified, `random_state=42`)

---

## Installation

```bash
pip install numpy pandas scikit-learn matplotlib seaborn \
            vaderSentiment textblob beautifulsoup4 \
            tensorflow gensim
```

---

## How to Run

Open each model's notebook in Jupyter and use **Kernel → Restart & Run All**.
Each notebook saves its results automatically to its own `results/` folder.

---

## Assignment 1 Baseline

The baseline from Assignment 1 was a 1-hidden-layer NumPy MLP using only 2 features
(VADER compound + TextBlob polarity), trained with vanilla SGD — achieving ~71% test
accuracy. All three Assignment 2 models improve on this baseline.
