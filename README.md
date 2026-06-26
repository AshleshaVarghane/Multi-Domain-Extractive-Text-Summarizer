#  Multi-Domain Extractive Summarizer

> Automatic Summarization Across Diverse Text Domains

A hybrid NLP-powered text summarization system that extracts the most important sentences from long-form documents across multiple domains — news, research, legal, medical, and more — using a combination of graph-based ranking, statistical scoring, and machine learning classification.

---


##  Project Info

| Field | Details |
|---|---|
| **Institution** | Shri Ramdeobaba College of Engineering & Management, Nagpur |
| **Program** | B.Tech – Computer Science & Engineering (Data Science) |
| **Semester** | VI Semester |
| **Authors** | Samiksha Shete (C2-28), Ashlesha Varghane (C4-69) |
| **GitHub** | [samshete05/Multi-domain-Extractive-Summarizer](https://github.com/samshete05/Multi-domain-Extractive-Summarizer) |

---

##  Overview

In the era of information explosion, vast amounts of text are generated daily across healthcare, finance, education, legal, and news domains. Processing this manually is time-consuming and inefficient.

This project builds a **Multi-Domain Extractive Summarizer** that:
- Identifies and extracts the most relevant sentences from any document
- Works across diverse domains without domain-specific tuning
- Maintains readability and coherence in summaries
- Deploys through a user-friendly Streamlit web interface

### What is Extractive Summarization?
Unlike **abstractive summarization** (which rewrites content in new words), **extractive summarization** selects and returns the most important original sentences directly from the document. This preserves factual accuracy and avoids hallucination.

---

##  Project Architecture

```
Input Document
      │
      ▼
┌─────────────────────┐
│   Preprocessing     │  → Sentence tokenization, stopword removal,
│                     │    lowercasing, lemmatization
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Feature Extraction │  → TF-IDF Vectorization (5000 features)
│                     │    + TextRank (graph-based) + BM25 (retrieval)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  ML Classification  │  → Logistic Regression vs Naive Bayes
│                     │    Best model auto-selected & saved as .pkl
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Sentence Ranking   │  → Hybrid score: 0.6×TextRank + 0.4×BM25
│  & Summary Output   │    Top-N sentences returned as summary
└─────────────────────┘
         │
         ▼
   Streamlit Web App
```

---

##  Methodology

### 1. Data Sources
| Dataset | Domain | Description |
|---|---|---|
| **CNN/DailyMail** | News | Articles with human-written highlight summaries |
| **XSum** | News (BBC) | Articles with single-sentence abstractive summaries |

Both datasets are stored as `.parquet` files and combined for training.

### 2. Preprocessing
- Sentence tokenization using NLTK's `sent_tokenize`
- Stopword removal (English stopwords via NLTK)
- Lowercasing and whitespace normalization
- Label assignment: `1` if sentence appears in summary, `0` otherwise

### 3. Feature Extraction
- **TF-IDF Vectorizer** with `max_features=5000` — converts sentences into numerical vectors based on word importance
- **TextRank** — graph-based ranking where sentences are nodes and cosine similarity scores are edges; PageRank determines importance
- **BM25Plus** — retrieval-based scoring that ranks sentences by keyword relevance to the full document

### 4. Hybrid Scoring Formula
```
Final Score = 0.6 × TextRank Score + 0.4 × BM25 Score
```

### 5. ML Models Compared
| Model | Strengths | Weaknesses |
|---|---|---|
| **Logistic Regression**  | Handles feature correlation, calibrated probabilities, better accuracy | Slower to train |
| **Naive Bayes** | Fast, simple, works with small data | Assumes feature independence (unrealistic for text) |

**Winner: Logistic Regression** — selected automatically based on accuracy on the test split.

---

## 📁 Project Structure

```
Multi-domain-Extractive-Summarizer/
│
├── src/
│   └── utils/
│       ├── constant.py          # Algorithm name constants
│       └── summarizer.py        # Core TextRank + BM25 hybrid logic
│
├── data/
│   ├── cnn/                     # CNN/DailyMail parquet files
│   └── xsum/                    # XSum parquet files
│
├── model/
│   ├── summarizer.pkl           # Pickled SummarizerModel (rule-based)
│   └── trained_model.pkl        # Pickled (best ML model + vectorizer)
│
├── app.py                       # Streamlit web application
├── main.py                      # CLI entry point for quick testing
├── train.py                     # Model training + evaluation script
├── create_model.py              # Saves rule-based summarizer as pickle
├── check_data.py                # Data inspection utility
└── requirements.txt             # Python dependencies
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip

### 1. Clone the Repository
```bash
git clone https://github.com/samshete05/Multi-domain-Extractive-Summarizer.git
cd Multi-domain-Extractive-Summarizer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data
```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
```

---

##  Usage

### Train the Model
```bash
python train.py
```
This loads CNN/DailyMail and XSum datasets, trains both Logistic Regression and Naive Bayes models, compares accuracy, and saves the best model to `model/trained_model.pkl`.

### Run the Web App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`, paste any document, and get a summary instantly.

### Quick CLI Test
```bash
python main.py
```
Runs the hybrid TextRank + BM25 summarizer on a sample paragraph and prints the summary to the terminal.

### Save the Rule-Based Model
```bash
python create_model.py
```
Wraps the `summarize_text()` function in a `SummarizerModel` class and saves it as `model/summarizer.pkl`.

### Inspect Dataset
```bash
python check_data.py
```
Prints column names and the first 5 rows of the CSV dataset for verification.

---

##  Dependencies

```
nltk
networkx
numpy
pandas
scikit-learn
rank_bm25
streamlit
pyarrow
openpyxl
```

Install all with:
```bash
pip install -r requirements.txt
```

---

##  Results

### Models Trained
- Logistic Regression
- Naive Bayes (MultinomialNB)

Both trained on TF-IDF features (5000 dimensions) from combined CNN/DailyMail + XSum datasets.

### Best Model
**Logistic Regression** — automatically selected and saved based on highest accuracy on the 80/20 train-test split.

### Why Logistic Regression is More Reliable
1. Does **not** assume feature independence (unlike Naive Bayes)
2. Produces well-calibrated `predict_proba()` scores used for sentence ranking
3. Consistently outperforms Naive Bayes on dense TF-IDF vectors
4. Better generalization across diverse domains

---

##  Web Application

The Streamlit app (`app.py`) provides:
- A text area to paste any document
- A slider to control the number of summary sentences (1–10)
- A "Summarize" button that runs the ML model
- Instant display of the extracted summary

**How it works internally:**
1. Input text is split into sentences
2. Each sentence is cleaned and vectorized using the saved TF-IDF vectorizer
3. The trained ML model assigns a summary-probability score to each sentence
4. Top-N sentences (by probability) are returned as the summary

---

##  Future Enhancements

- **Adaptive domain weighting** — automatically detect and weight features per domain
- **Cross-lingual summarization** — support for non-English documents
- **Abstractive summarization integration** — combine extractive + generative approaches
- **Reinforcement learning optimization** — use ROUGE-based rewards to improve summary quality
- **API endpoint** — expose summarizer as a REST API for integration into other apps

---

##  License

This project was developed as an academic submission for Shri Ramdeobaba College of Engineering & Management, Nagpur. All rights reserved by the authors.

---

##  Acknowledgements

- [CNN/DailyMail Dataset](https://huggingface.co/datasets/cnn_dailymail)
- [XSum Dataset](https://huggingface.co/datasets/xsum)
- [NLTK](https://www.nltk.org/)
- [scikit-learn](https://scikit-learn.org/)
- [NetworkX](https://networkx.org/)
- [rank-bm25](https://github.com/dorianbrown/rank_bm25)
- [Streamlit](https://streamlit.io/)

