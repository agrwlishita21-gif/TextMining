# TextMining
# TextMining


## Topic Modelling: BERTopic 
Note: The file `lid.176.bin` is not included due to GitHub size limits.
Please download it from: https://fasttext.cc/docs/en/language-identification.html and place it in the `topic_modelling/BERTopic/` directory.

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


