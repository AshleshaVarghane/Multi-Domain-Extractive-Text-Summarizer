# train.py

import pandas as pd
import os
import glob
import numpy as np
import nltk
import pickle

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

nltk.download("punkt")
nltk.download("stopwords")

print("=== Loading datasets ===")

# ---- CNN DailyMail ----
cnn = pd.read_parquet("data/cnn/augmented_cnn_dailymail.parquet")
cnn = cnn[['article','highlights']].dropna()

# ---- XSum ----
xsum = pd.read_parquet("data/xsum/augmented_xsum.parquet")
xsum = xsum.rename(columns={'document':'article','summary':'highlights'})
xsum = xsum[['article','highlights']].dropna()

# ---- COMBINE ----
df = pd.concat([cnn, xsum], ignore_index=True)

df = df.dropna()
df['article'] = df['article'].astype(str)
df['highlights'] = df['highlights'].astype(str)

# SAMPLE for faster iteration during development. Set SAMPLE_LIMIT=None to use full dataset.
SAMPLE_LIMIT = 1000
total_rows = len(df)
if SAMPLE_LIMIT is not None and total_rows > SAMPLE_LIMIT:
    print(f"Dataset large ({total_rows} rows). Sampling {SAMPLE_LIMIT} rows for a faster run.")
    df = df.sample(SAMPLE_LIMIT, random_state=42)
else:
    print("Total training dataset size:", total_rows)

# ---- Preprocessing ----
stop_words = set(stopwords.words('english'))

def preprocess(sent):
    sent = sent.lower().strip()
    words = sent.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

print("=== Creating training samples ===")

texts = []
labels = []

for _, row in df.iterrows():
    article = row['article']
    summary_text = row['highlights']
    sentences = sent_tokenize(article)

    for sent in sentences:
        cleaned = preprocess(sent)
        if cleaned == "":
            continue
        texts.append(cleaned)
        labels.append(1 if sent in summary_text else 0)

print("Total sentences:", len(texts))
print("Positive summary sentences:", sum(labels))

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)
y = np.array(labels)

print("=== Training Models ===")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(max_iter=500),
    "NaiveBayes": MultinomialNB()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"=== {name} Results ===")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap="Blues")
    os.makedirs("model", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"model/confusion_{name}.png")
    plt.close()
def load_parquet_dir(dirpath):
    pattern = os.path.join(dirpath, "*.parquet")
    files = glob.glob(pattern)
    if not files:
        print(f"Warning: no parquet files found in {dirpath}")
        return pd.DataFrame(columns=["article", "highlights"])

    path = files[0]
    print(f"Loading parquet: {path}")
    df_local = pd.read_parquet(path)

    def _find(cols, candidates):
        for c in cols:
            if any(k.lower() in c.lower() for k in candidates):
                return c
        return None

    cols = list(df_local.columns)
    article_col = _find(cols, ["article", "document", "text", "content", "body"])
    summary_col = _find(cols, ["highlights", "summary", "abstract", "short_summary"])

    if article_col is None:
        print(f"Warning: no article-like column in {path}; available columns: {cols[:20]}")
        return pd.DataFrame(columns=["article", "highlights"])

    rename_map = {article_col: "article"}
    if summary_col:
        rename_map[summary_col] = "highlights"

    df_local = df_local.rename(columns=rename_map)
    if "highlights" not in df_local.columns:
        df_local["highlights"] = ""

    return df_local[["article", "highlights"]].dropna(subset=["article"])


cnn = load_parquet_dir("data/cnn")
for m, score in results.items():
    print(m, "=", score)
xsum = load_parquet_dir("data/xsum")
best = max(results, key=results.get)
best_model = models[best]

print("\nBest model:", best)

with open("model/trained_model.pkl","wb") as f:
    pickle.dump((best_model, vectorizer), f)

print("Training complete and model saved!")