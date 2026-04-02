# TextMining

This repo contains the final notebook workflow for lyric sentiment and emotion analysis:

- lexicon-based sentiment with VADER
- feature engineering with NRC and VAD lexicons
- linear supervised baselines with Logistic Regression and Linear SVM
- a transformer-based emotion classifier fine-tuned on lyric emotion labels

Main notebook:

- [FinalSentimentAnalysis.ipynb](/Users/sheen/Desktop/work/SMU/y3s1/textmining/TextMining/FinalSentimentAnalysis.ipynb)

## What Is Included

Committed:

- the final notebook with setup, preprocessing, model training, and cleaned result summaries
- `NRC-VAD-Lexicon.txt` so the notebook can run without fetching that lexicon separately
- `llm_annotated_8500.csv`, which is the existing LLM-labeled emotion dataset used by the supervised models
- `transformer_song_level_predictions.csv`, which stores the final transformer test predictions

Not committed:

- `songs.csv` and `artists.csv` from Kaggle
- multi-GB intermediate CSVs and sparse matrices
- transformer checkpoints and model weights

Those files are intentionally excluded because they are too large for a normal GitHub repo and can be regenerated from the notebook.

## Data Source

Raw Spotify lyrics dataset:

- [Kaggle: 550K Spotify Songs, Audio, Lyrics, and Genres](https://www.kaggle.com/datasets/serkantysz/550k-spotify-songs-audio-lyrics-and-genres)

## Reproducibility

1. Open the notebook.
2. Run the bootstrap/setup cells near the top.
   They install missing packages, download the Kaggle dataset, and prepare any missing local files.
3. Keep `llm_annotated_8500.csv` in the repo root.
   This allows the supervised models and transformer section to run without a Groq API key.
4. Run the notebook cells in order.

Notes:

- The file name `llm_annotated_8500.csv` is a legacy name; the committed file contains 20,500 labeled songs.
- The notebook is set up to skip Groq labeling if that file already exists.
- The transformer model weights are not committed; rerunning the transformer section regenerates them locally.

## Final Comparison

- Logistic Regression: test accuracy `0.68`, macro F1 `0.66`
- Linear SVM: slightly below Logistic Regression on validation, so not selected as the final linear baseline
- Transformer (`j-hartmann/emotion-english-distilroberta-base`): song-level test accuracy `0.71`, macro F1 `0.68`

## Limitation

The supervised models were trained and evaluated on LLM-generated emotion labels rather than human gold labels. This makes the setup useful for weakly supervised model comparison, but the reported metrics should be interpreted as agreement with the labeling model, not as definitive ground-truth emotion performance.
