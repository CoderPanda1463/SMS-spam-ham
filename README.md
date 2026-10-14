# SMS Spam Classification using SVM

A machine learning project that classifies SMS messages as **spam** or **ham** (legitimate) using a Linear Support Vector Machine (LinearSVC) with TF-IDF text features and a few engineered numeric features.

**Dataset:** [SMS Spam Collection Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) — download `spam.csv`.

## Features

- Exploratory data analysis: class distribution, word-count histograms, spam/ham word clouds
- Text preprocessing: lowercasing, URL/email/digit/punctuation removal, stopword removal, WordNet lemmatization, duplicate removal
- Feature engineering:
  - `msg_len` — length of the cleaned message
  - `digit_count` — number of digits in the original message
- Multi-feature model: `ColumnTransformer` combining TF-IDF (unigrams + bigrams, top 5000 features) with a scaled numeric features
- Baseline model: TF-IDF + LinearSVC on text only
- Hyperparameter tuning of `C` (0.1, 1, 10) with `GridSearchCV` (5-fold)
- 5-fold cross-validation
- Evaluation: accuracy, precision, recall, F1, classification report, confusion matrix, ROC curve and AUC
- `predict_sms(text)` helper for classifying new messages

## Project Structure

```
.
├── SMS_Spam_SVM.ipynb   # Main notebook
├── spam.csv             # Dataset (download from Kaggle)
└── README.md
```

## Requirements

- Python 3.8+
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- nltk
- wordcloud

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud
```

## Getting Started

### Google Colab (recommended)

1. Open `SMS_Spam_SVM.ipynb` in Google Colab.
2. Run the install and import cells.
3. When prompted by the upload cell, upload `spam.csv` from Kaggle.
4. Run the remaining cells in order.

### Local

1. Download `spam.csv` from the Kaggle link above and place it next to the notebook.
2. Replace the Colab upload cell (`from google.colab import files ...`) with nothing — the next cell reads `spam.csv` directly.
3. Launch Jupyter and run all cells:

```bash
jupyter notebook SMS_Spam_SVM.ipynb
```

## Workflow

1. **Load data** — read `spam.csv` (latin-1 encoding), keep columns `v1` (label) and `v2` (message).
2. **EDA** — inspect shape, nulls, duplicates, class balance; visualize lengths and word clouds.
3. **Preprocess** — clean text, drop duplicates, map `ham → 0`, `spam → 1`.
4. **Engineer features** — add `msg_len` and `digit_count`.
5. **Split** — 80/20 stratified train/test split (`random_state=42`).
6. **Train** — fit TF-IDF + LinearSVC pipeline.
7. **Tune** — grid search over `C`.
8. **Validate** — cross-validation on the best estimator.
9. **Evaluate** — metrics, confusion matrix, ROC curve.
10. **Predict** — classify new messages with `predict_sms()`.

## Usage

```python
predict_sms("Congratulations! You have won a free iPhone. Click here now.")  # Spam
predict_sms("Hi, are we meeting tomorrow at 3 PM?")                          # Ham
```

## Notes

- The notebook contains two versions of several steps (the original text-only pipeline and the refined multi-feature pipeline with `msg_len` and `digit_count`). Run the cells in order, or remove the redundant earlier versions, so the later cells use the intended model. In particular, the text-only train/test split and pipeline cells overwrite the multi-feature ones if run afterward.
- `LinearSVC` has no `predict_proba`, so the ROC curve uses `decision_function` scores.

## License

Dataset license and terms are governed by the original Kaggle/UCI source. Add your own license for the code as needed.
