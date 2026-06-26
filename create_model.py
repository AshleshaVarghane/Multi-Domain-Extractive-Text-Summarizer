import pickle
from src.utils.summarizer import summarize_text

class SummarizerModel:

    def predict(self, text, num_sentences=3):
        return summarize_text(text, num_sentences)

model = SummarizerModel()

with open("model/summarizer.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")