# app.py

import streamlit as st
import pickle
from nltk.tokenize import sent_tokenize

st.title("🧠 Multi‑Domain Extractive Summarizer")

with open("model/trained_model.pkl","rb") as f:
    model, vectorizer = pickle.load(f)

text = st.text_area("Enter Document", height=250)
num = st.slider("Summary Sentences", 1, 10, 3)

if st.button("Summarize"):
    if text.strip()=="":
        st.error("Enter some text!")
    else:
        sentences = sent_tokenize(text)
        cleaned = [s.lower().strip() for s in sentences]

        X = vectorizer.transform(cleaned)
        probs = model.predict_proba(X)[:,1]

        ranked = sorted(zip(probs,sentences), reverse=True)
        summary = " ".join([s for _,s in ranked[:num]])

        st.subheader("Summary:")
        st.write(summary)