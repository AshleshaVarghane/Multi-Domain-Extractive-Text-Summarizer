# src/utils/summarizer.py
import numpy as np
import networkx as nx
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Plus
import nltk

nltk.download("punkt")

def summarize_text(text, num_sentences=3):

    sentences = sent_tokenize(text)

    if len(sentences) <= num_sentences:
        return text

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(sentences)

    similarity_matrix = cosine_similarity(tfidf_matrix)

    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)

    tokenized = [s.split() for s in sentences]
    bm25 = BM25Plus(tokenized)
    doc_tokens = [tok for sent in tokenized for tok in sent]
    bm25_scores = bm25.get_scores(doc_tokens) if doc_tokens else [0] * len(sentences)

    final_scores = []
    for i in range(len(sentences)):
        score = 0.6 * scores[i] + 0.4 * bm25_scores[i]
        final_scores.append((score, sentences[i]))

    ranked = sorted(final_scores, reverse=True)
    summary = " ".join([s for _, s in ranked[:num_sentences]])

    return summary