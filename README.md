# TextMining
# TextMining
# Song Lyrics Emotion Classification — NLP Pipeline


### 1. `FinalSentimentAnalysis.ipynb` — Feature Engineering & Classical ML

End-to-end pipeline from raw Spotify data to a trained SVM classifier.

#### Pipeline Overview

**Phase 1 — Data Loading & Cleaning**
- Loads `songs.csv` (550,622 songs) and `artists.csv` (71,440 artists)
- Removes songs with timestamps in lyrics, repeated lines, and non-English songs
- Applies slang normalization (~150 contractions/informal terms mapped to standard English)
- Filters songs with >30% non-English words using NLTK dictionary frequency analysis
- Final clean dataset: **520,589 English songs**

**Phase 2 — Sentiment & Emotion Feature Engineering**

| Feature Set | Method | Output Features |
|---|---|---|
| VADER Sentiment | Line-by-line compound scoring | `vader_compound`, `vader_sentiment` |
| NRC Emotion Lexicon | 10-dimension emotion extraction | `nrc_joy`, `nrc_sadness`, `nrc_anger`, `nrc_fear`, `nrc_positive`, `nrc_negative`, `nrc_anticipation`, `nrc_trust`, `nrc_surprise`, `nrc_disgust` |
| VAD Scores | Valence-Arousal-Dominance lexicon (19,971 words) | `vad_valence`, `vad_arousal`, `vad_dominance` |
| Text Patterns | Repetition ratio, negation count, exclamation ratio | `repetition_ratio`, `negation_count`, `exclamation_ratio` |
| TF-IDF | 30,000 features, unigrams + bigrams, sublinear TF | Sparse matrix |

Combined feature matrix shape: **(520,589 × 30,019)**

**Phase 3 — LLM Annotation (Ground Truth Labels)**
- Samples 2,050 songs per genre across 10 genres (20,500 total)
- Uses **Groq `llama-3.3-70b-versatile`** to classify into 6 emotions: Happy, Sad, Angry, Energetic, Romantic, Nostalgia
- Produces `llm_annotated_8500.csv` with 8,500 validated labels
- Final training set (4 main emotions): **11,291 songs**

**Phase 4 — Supervised Classification**

| Model | Val Macro F1 | Test Accuracy | Test Macro F1 |
|---|---|---|---|
| Logistic Regression | 0.666 | — | — |
| **Linear SVM (best)** | **0.688** | **68%** | **0.66** |

Per-class test F1 (SVM): Sad=0.78, Energetic=0.67, Happy=0.63, Angry=0.58

#### Key Files Produced
| File | Description |
|---|---|
| `songs_ready_for_classifier.csv` | Cleaned dataset with all engineered features |
| `llm_annotated_8500.csv` | LLM emotion labels for 8,500 songs |
| `tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `scaler.pkl` | StandardScaler for dense features |
| `X_combined.npz` | Full sparse+dense feature matrix (703 MB) |

---

### 2. `BERT_SentimentAnalysis.ipynb` — Transformer Fine-Tuning

Fine-tunes a pre-trained RoBERTa model on the same emotion classification task, using chunk-level inference for long lyrics.

#### Pipeline Overview

**Step 1 — Data Preparation**
- Loads `songs_ready_for_classifier.csv` and `llm_annotated_8500.csv`
- Filters to 4 emotion classes: Happy, Sad, Angry, Energetic
- Caps at 5,000 songs per label (stratified sampling)
- Final label distribution: Sad=4,044 · Energetic=3,330 · Angry=2,404 · Happy=1,499

**Step 2 — Lyrics Chunking**
Long lyrics are split into overlapping chunks before tokenization:

| Parameter | Value |
|---|---|
| Words per chunk | 180 |
| Stride (overlap) | 120 |
| Minimum chunk words | 40 |

Song-level splits (stratified):

| Split | Songs | Chunks |
|---|---|---|
| Train | 9,021 | 21,038 |
| Validation | 1,128 | 2,673 |
| Test | 1,128 | 2,597 |

**Step 3 — Model Fine-Tuning**
- Base model: `j-hartmann/emotion-english-distilroberta-base`
- Classifier head re-initialized for 4 output classes (original had 7)
- Max token length: 256

| Hyperparameter | Value |
|---|---|
| Learning rate | 2e-5 |
| Batch size | 8 |
| Epochs | 10 |
| Weight decay | 0.01 |
| Best model selection | Max macro F1 on validation |

**Step 4 — Song-Level Inference**
Chunk-level softmax probabilities are averaged per song to produce a final prediction:

**Test Results (song-level)**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Happy | 0.84 | 0.57 | 0.68 | 150 |
| Sad | 0.74 | 0.91 | 0.81 | 404 |
| Angry | 0.64 | 0.53 | 0.58 | 241 |
| Energetic | 0.74 | 0.73 | 0.73 | 333 |
| **Overall** | | **0.73** | **0.72** | **1,128** |

Macro F1: **0.70** | Weighted F1: **0.72** | Accuracy: **73%**

**Step 5 — Demo Inference**
The notebook includes an end-to-end demo on an unseen song, displaying per-label confidence scores:

```
Energetic   : 0.713  █████████████████████
Sad         : 0.176  █████
Angry       : 0.100  ███
Happy       : 0.011
→ Energetic (71.3% confidence)
```

#### Key Files Produced
| File | Description |
|---|---|
| `transformer_training_data.csv` | Labeled training set fed to the model |
| `transformer_emotion_model/` | Saved fine-tuned model + tokenizer |
| `transformer_song_level_predictions.csv` | Song-level probabilities and predicted labels |
| `bert_predictions.csv` | Ground truth vs. predicted labels for test set |

---

## Model Comparison

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Logistic Regression | ~67% | 0.65 | — |
| Linear SVM | 68% | 0.66 | — |
| **DistilRoBERTa (fine-tuned)** | **73%** | **0.70** | **0.72** |

The transformer model improves over the classical SVM by ~5 percentage points on both accuracy and macro F1, with the largest gains on the Energetic and Happy classes.

## Resource Downloads

### VADER Lexicon
VADER is bundled with the `vaderSentiment` package and also available via NLTK:
```bash
pip install vaderSentiment

# Or via NLTK:
pip install nltk
python -c "import nltk; nltk.download('vader_lexicon')"
```

### NRC Emotion Lexicon (NRCLex)
Used for the 10 emotion dimensions (joy, sadness, anger, etc.):
```bash
pip install nrclex
```
NRCLex ships with its own lexicon file — no separate download needed.

### NRC VAD Lexicon
The `NRC-VAD-Lexicon.txt` file (Valence-Arousal-Dominance scores) must be downloaded separately from the NRC Canada website:
> https://saifmohammad.com/WebPages/nrc-vad.html

## Data Requirements

- `songs.csv` — Raw Spotify song dataset with lyrics and audio features
- `artists.csv` — Artist metadata
- `llm_annotated_8500.csv` — Pre-generated LLM labels (produced by `FinalSentimentAnalysis.ipynb`)

Run `FinalSentimentAnalysis.ipynb` first to generate all intermediate files before running `BERT_SentimentAnalysis.ipynb`.


## Topic Modelling: BERTopic 
Note: The file `lid.176.bin` is not included due to GitHub size limits.
Please download it from: https://fasttext.cc/docs/en/language-identification.html and place it in the `topic_modelling/BERTopic/` directory.

Additionally, ensure that the dataset `songs_with_vader.csv` is placed under the `TMdata/` folder in the project root directory.

**Note:** BERTopic was performed on a sampled subset of 20,000 songs due to computational constraints.

### BERTopic Experiments
#### 1. Segmentation & Hyperparameter Tuning

The following experiments evaluate the impact of chunk size and `n_neighbors` on topic quality.

| Version | Chunk Size (min words) | n_neighbors | Coherence Score | Topic Diversity | No. of Topics |
|--------|------------------------|------------|----------------|----------------|--------------|
| bertopic_model_v1.ipynb | 6 | 80 | 0.3257 | 0.8314 | 328 |
| bertopic_model_v2.ipynb | 10 | 80 | 0.4230 | 0.8000 | 106 |
| bertopic_model_v5.ipynb | 12 | 40 | 0.4543 | 0.7987 | 78 |
| bertopic_model_v6.ipynb | 12 | 60 | 0.4554 | 0.8053 | 75 |
| **bertopic_model_v3.ipynb (final)** | 12 | 80 | **0.4616** | **0.8091** | **66** |

With chunk size fixed at 12 words, increasing `n_neighbors` (40 → 80) improved topic coherence and stability, with 80 producing the best overall performance.

---

#### 2. Stopword Removal Impact
This experiment evaluates the effect of preprocessing on topic quality.

| Version | Preprocessing | Coherence Score | Topic Diversity | No. of Topics |
|--------|--------------|----------------|----------------|--------------|
| **bertopic_model_v3.ipynb (final)** | Stopword removal | **0.4616** | **0.8091** | **66** |
| bertopic_model_v4.ipynb | No stopword removal | 0.3633 | 0.7293 | 297 |

Retaining stopwords introduced noise and resulted in fragmented, less interpretable topics. Removing stopwords significantly improved coherence and reduced the number of topics.

## Topic Modelling: LDA & NMF

> **Note:** The preprocessed dataset `songs_preprocessed.parquet` must be placed in the `data/` directory before running. Models and results are saved to `lda/` and `nmf/` directories respectively.

**Note:** NMF was performed on the full dataset of **548,698 songs**, while LDA was performed on a subset due to the time it took.

---

### LDA (Latent Dirichlet Allocation)

LDA was tuned from K=20 to K=100 (step 5) using Gensim, evaluated on **Coherence Score (Cv)** and **Topic Diversity**.

#### Hyperparameter Tuning

| Parameter | Value |
|---|---|
| Topic range tested | K = 20 to K = 100 (step 5) |
| Passes | 10 |
| Alpha | auto |
| Eta | auto |
| Chunk size | 2,000 |
| Random state | 42 |

#### Best LDA Result

| Metric | Score |
|---|---|
| Coherence Score (Cv) | 0.4745 |
| Diversity | 0.935 |

#### Key Files Produced

| File | Description |
|---|---|
| `lda/lda_dictionary.dict` | Fitted Gensim dictionary |
| `lda/lda_corpus.mm` | Serialised Gensim corpus |
| `lda/lda_tuning_results.csv` | Coherence & diversity scores per K |
| `lda/tuning_models/lda_model_k{k}` | Saved LDA models for each K |

---

### NMF (Non-negative Matrix Factorization)

NMF was tuned from K=20 to K=100 (step 5) using scikit-learn with TF-IDF vectorisation, evaluated on **Coherence Score (Cv)** and **Topic Diversity**.

#### Hyperparameter Tuning

| Parameter | Value |
|---|---|
| Topic range tested | K = 20 to K = 100 (step 5) |
| Vectoriser | TF-IDF (max_df=0.90, min_df=10, ngram_range=(1,2)) |
| NMF solver | `cd` (coordinate descent) |
| Random state | 42 |

#### Best NMF Result

| Metric | Score |
|---|---|
| Coherence Score (Cv) | 0.6042 |
| Topic Diversity | 0.8840 |

#### Key Files Produced

| File | Description |
|---|---|
| `nmf/tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `nmf/nmf_model.pkl` | Best fitted NMF model |
| `nmf/nmf_tuning_results.csv` | Coherence & diversity scores per K |

---

### LDA vs NMF Comparison

| Model | Coherence Score (Cv) | Diversity | Notes |
|---|---|---|---|
| LDA | 0.4745 | 0.935 | Better probabilistic interpretability |
| NMF | 0.6042 | 0.8840 | Higher Choerence; more semantically focused topics |

> NMF provided better results at initial testing from K = 20 to K = 100. Parameters such as ngrams, max_df, and min_df were tuned to improve performance, achieving the final Coherence Score of 0.6042. As we focused on Coherence Score, we chose NMF as our final model.
