# Application Review Sentiment & Urgency Analysis 

A multi-task review intelligence system for Google Play Store app reviews that automatically triages user feedback by predicting sentiment, urgency, priority, and detected aspects — with attention-based explainability.

---
## Tech Stack

<div align="center">

# Tech Stack

<br>

<img src="https://img.shields.io/badge/Python-111827?style=for-the-badge&logo=python&logoColor=yellow" />
<img src="https://img.shields.io/badge/PyTorch-111827?style=for-the-badge&logo=pytorch&logoColor=red" />
<img src="https://img.shields.io/badge/scikit--learn-111827?style=for-the-badge&logo=scikit-learn&logoColor=orange" />

<br><br>

<img src="https://img.shields.io/badge/FastAPI-111827?style=for-the-badge&logo=fastapi&logoColor=00C7B7" />
<img src="https://img.shields.io/badge/Streamlit-111827?style=for-the-badge&logo=streamlit&logoColor=ff4b4b" />
<img src="https://img.shields.io/badge/Pandas-111827?style=for-the-badge&logo=pandas&logoColor=white" />
<img src="https://img.shields.io/badge/NumPy-111827?style=for-the-badge&logo=numpy&logoColor=4DA6FF" />

<br><br>

<img src="https://img.shields.io/badge/NLTK-111827?style=for-the-badge" />
<img src="https://img.shields.io/badge/GloVe_Embeddings-111827?style=for-the-badge" />
<img src="https://img.shields.io/badge/Uvicorn-111827?style=for-the-badge" />
<img src="https://img.shields.io/badge/Pydantic-111827?style=for-the-badge&logo=pydantic&logoColor=E92063" />

</div>

## What This System Does

| Output | Values |
|--------|--------|
| Sentiment | Negative / Neutral / Positive |
| Urgency | Low / Medium / High |
| Priority | Low / Medium / High / Critical |
| Aspects | UI, Performance, Battery, Network, Login/Auth, Payment, and more |
| Explanation | Top influential words from attention weights |

---

## Project Goal

Automatically triage app reviews so product and support teams can:

- Detect sentiment and urgency together
- Prioritize the most critical feedback first
- Understand which aspects of the app are being discussed
- Get simple explanations for model predictions
- Process both single reviews and CSV files at scale

---

## System Architecture

This is the end-to-end architecture of the full system — from raw review to API response.

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                               │
│                                                                     │
│   Streamlit UI (single review + CSV batch)   Postman / HTTP client  │
└───────────────────────────┬─────────────────────────────────────────┘
                            │  POST /analyze  { "text": "..." }
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FASTAPI LAYER                               │
│                                                                     │
│   GET  /        →  health check                                     │
│   POST /analyze →  validate (ReviewRequest) → pipeline.run()        │
│                    fallback on error → structured error response     │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       INFERENCE PIPELINE                            │
│                       src/pipeline.py                               │
│                                                                     │
│   1. TextPreprocessor   →  clean + tokenize + pad sequence          │
│   2. TF-IDF Vectorizer  →  sparse keyword feature vector            │
│   3. HybridModel        →  sentiment logits, urgency logits, attn   │
│   4. LabelEncoders      →  decode numbers → string labels           │
│   5. PriorityScorer     →  rule-based priority from sent + urg      │
│   6. AspectDetector     →  keyword match → aspect categories        │
│   7. Explainer          →  attention weights → top words            │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        RESPONSE PAYLOAD                             │
│                                                                     │
│   sentiment   → "Negative"                                          │
│   urgency     → "High"                                              │
│   priority    → "Critical"                                          │
│   aspects     → ["Performance", "Battery"]                          │
│   explanation → { top_words: [{"word": "crash", "score": 0.91}] }   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Model Progression

The repository documents a clear progression from a simple baseline to the final hybrid deep learning architecture.

### Model Comparison

| Model | Sentiment Accuracy | Urgency Accuracy | Notes |
|---|---:|---:|---|
| Baseline TF-IDF + Logistic Regression | — | — | Simplest reference model |
| Multi-Task LSTM | 0.6320 | 0.9125 | Shared sequence model |
| BiLSTM + Attention | 0.6687 | 0.9047 | Interpretable sequence model |
| **Hybrid Model** | **0.84** | **0.97** | Best overall performance |

---

### 1. Baseline Model

**File:** `src/baseline_model/train_baseline_rev+urg_model.py`

```
Review Text
    │
    ▼
TF-IDF Vectorizer (5,000 features)
    │
    ├──► Logistic Regression → Sentiment label
    │
    └──► Logistic Regression → Urgency label
```

- Two separate, independent classifiers
- No shared representation
- Fast to train, interpretable, but misses sequential context
- Artifacts saved under `models/baseline/`

---

### 2. Multi-Task LSTM

**File:** `src/lstm_model/train_multitask_lstm.py`

```
Review Text
    │
    ▼
Tokenizer (10,000 vocab, max len 50)
    │
    ▼
Embedding Layer (learned from scratch)
    │
    ▼
LSTM
    │
    ▼
Shared Fully Connected Layer
    │
    ├──► Sentiment Head → Negative / Neutral / Positive
    │
    └──► Urgency Head   → Low / Medium / High
```

- Single model, two output heads sharing one representation
- Introduces sequential learning over word order
- No bidirectional context, no attention
- Artifacts saved under `models/lstm/`

---

### 3. BiLSTM + Attention

**File:** `src/bilstm_attention_model/train_bilstm_attention_model.py`

```
Review Text
    │
    ▼
Tokenizer (10,000 vocab, max len 100)
    │
    ▼
GloVe 6B 100d Embedding (pre-trained, fine-tuned)
    │
    ▼
BiLSTM  ◄── reads text left-to-right AND right-to-left
    │
    ▼
Attention Layer  ◄── scores each word by importance
    │             returns: context vector + attention weights
    ▼
Shared Dense Layer
    │
    ├──► Sentiment Head → Negative / Neutral / Positive
    │
    └──► Urgency Head   → Low / Medium / High
```

- Bidirectional context improves understanding of word meaning
- Attention weights make word importance visible (used later for explainability)
- GloVe embeddings give the model pre-trained word knowledge
- Artifacts saved under `models/bilstm_attention_sen_urg/`

---

### 4. Hybrid Model (Final)

**Files:** `src/hybrid_model/train_hybrid_model.py` · `src/hybrid_model/hybrid_model.py`

This is the production model. It combines two parallel feature streams that capture complementary signals.

---

## Hybrid Model Architecture

```
                        Review Text
                            │
               ┌────────────┴────────────┐
               │                         │
               ▼                         ▼
     TextPreprocessor              TextPreprocessor
     (clean + tokenize)            (clean text only)
               │                         │
               ▼                         ▼
     Padded Token Sequence        TF-IDF Vectorizer
     (max len 100, vocab 10k)     (8,000 features)
               │                         │
               │                         │
    ┌──────────┘                         └──────────┐
    │         SEQUENCE BRANCH          LEXICAL BRANCH│
    │                                               │
    ▼                                               ▼
GloVe Embedding                          Linear Projection
(100d, pre-trained,                      (8,000 → 128)
 fine-tuned during training)                        │
    │                                               ▼
    ▼                                          ReLU activation
2-Layer BiLSTM                                      │
(hidden=128, bidirectional,                         │
 2 layers, dropout=0.3)                             │
output: 256 numbers per token                       │
    │                                               │
    ▼                                               │
Attention Layer                                     │
  score each token with Linear(256→1)               │
  apply softmax → weights                           │
  weighted sum → context vector (256d)              │
    │                                               │
    └──────────────┐      ┌─────────────────────────┘
                   │      │
                   ▼      ▼
              Concatenate (256 + 128 = 384d)
                      │
                      ▼
              Dense Layer (384 → 256)
              + ReLU + Dropout (0.4)
                      │
               ┌──────┴──────┐
               │             │
               ▼             ▼
       Sentiment Head    Urgency Head
       Linear(256→3)     Linear(256→3)
               │             │
               ▼             ▼
       Negative/Neutral/  Low/Medium/High
       Positive
```

### Why This Architecture Works

The hybrid design combines two signals that are individually limited:

| Stream | What it captures | What it misses |
|--------|------------------|----------------|
| BiLSTM + Attention | Word order, context, sentiment around negations ("not great") | Rare but critical keywords |
| TF-IDF | High-frequency crash/error/bug keywords | All sequential meaning |
| **Combined** | **Both context and keyword signal** | — |

This is the primary reason the hybrid achieves **0.84 sentiment accuracy** and **0.97 urgency accuracy** versus 0.63 for a plain LSTM.

---

## Preprocessing Pipeline

**File:** `src/preprocessing.py`

```
Raw Review Text
    │
    ▼  lowercase
    ▼  remove URLs
    ▼  expand contractions  (don't → do not)
    ▼  remove non-ASCII and special characters
    ▼  normalize whitespace
    ▼  POS tagging
    ▼  lemmatization  (crashing → crash)
    ▼  tokenize
    ▼  pad to fixed length
    │
    ▼
Clean Sequence (for model) + Clean Text (for TF-IDF)
```

### Class Balancing

The hybrid training script applies **random oversampling** before training:

- Positive reviews are kept as the reference class
- Negative and Neutral reviews are upsampled to match
- This prevents the model from ignoring minority sentiment classes

---

## Explainability, Aspect Detection, and Priority Scoring

### Attention-Based Explanation

**File:** `src/explain.py`

```
Attention weights (from model forward pass)
    │
    ▼
Map weight[i] → word[i]  (skip padding tokens)
    │
    ▼
Sort by score descending
    │
    ▼
Return top-K words with scores

Example output:
  {"word": "crash",   "score": 0.91}
  {"word": "freezes", "score": 0.74}
```

### Aspect Detection

**File:** `src/aspect.py`

Rule-based keyword matcher. Scans clean text for category keywords:

| Aspect | Example keywords |
|--------|-----------------|
| UI | layout, button, screen, design |
| Performance | slow, lag, freeze, crash |
| Battery | drain, battery, power |
| Network | wifi, connection, offline |
| Login/Auth | login, password, account |
| Payment | payment, purchase, refund |
| Ads | ads, popup, advertisement |
| Bug | bug, glitch, broken, error |
| Security | privacy, data, hack |
| ... | and more |

### Priority Scoring

**File:** `src/priority.py`

```
Sentiment + Urgency
       │
       ▼
Score-based rules:
  Negative  → +score
  Neutral   → +smaller score
  High urg  → +large score
  Medium urg→ +moderate score
       │
       ▼
Threshold → Low / Medium / High / Critical
```

---

## Inference Pipeline

**File:** `src/pipeline.py`

```
Input text
    │
    ├─► TextPreprocessor.text_to_sequence()  →  padded seq + clean text
    │
    ├─► tfidf.transform(clean_text)          →  TF-IDF vector
    │
    ├─► HybridModel.forward(seq, tfidf)      →  sent_logits, urg_logits, attn
    │
    ├─► sent_encoder.inverse_transform()     →  "Negative"
    ├─► urg_encoder.inverse_transform()      →  "High"
    │
    ├─► PriorityScorer.compute()             →  "Critical"
    ├─► AspectDetector.detect()              →  ["Performance", "Battery"]
    └─► Explainer.explain()                  →  [{"word": "crash", "score": 0.91}]
```

---

## API and Fallback Behavior

**File:** `src/api/app.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check — returns `{"message": "API is running"}` |
| `/analyze` | POST | Accepts `{"text": "..."}`, returns full analysis |

**Fallback behavior** — if inference fails for any reason, the API returns a structured response instead of crashing:

```json
{
  "sentiment": "Unknown",
  "urgency": "Low",
  "priority": "Low",
  "aspects": ["General Issue"],
  "explanation": {}
}
```

---

## Streamlit App and CSV Support

**File:** `app/streamlit_app.py`

### Single Review Workflow

```
User types review → POST /analyze → display results
  - Sentiment badge
  - Urgency badge
  - Priority badge
  - Detected aspects
  - Top explanation words
```

### CSV Batch Workflow

```
Upload CSV (must have "reviews" column)
    │
    ▼
For each row → POST /analyze → collect result
    │ (live progress bar)
    ▼
Display processed table
    │
    ▼
Download as processed_reviews.csv
```

---

## Saved Model Artifacts

```
models/
├── hybrid/                    ← production model
│   ├── hybrid_model.pt
│   ├── tokenizer.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── sent_encoder.pkl
│   └── urg_encoder.pkl
│
├── bilstm_attention_sen_urg/  ← generation 3
├── lstm/                      ← generation 2
└── baseline/                  ← generation 1
```

---

## Project Structure

```
Data/
  embedding/         ← GloVe 6B 100d
  playstore/         ← raw review data
  processed/         ← playstore_clean.csv

models/
  hybrid/
  bilstm_attention_sen_urg/
  lstm/
  baseline/

outputs/
  reports/           ← classification reports per model generation

src/
  baseline_model/
  lstm_model/
  bilstm_attention_model/
  hybrid_model/
  api/
  pipeline.py
  preprocessing.py
  explain.py
  aspect.py
  priority.py

app/
  streamlit_app.py
```

---

## How to Run

### 1. Start the API

```bash
uvicorn src.api.app:app --reload
```

### 2. Start the Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

### 3. Use the standalone pipeline

```bash
python pipeline.py
```

---

## Author
Aryan Kumar Yadav
