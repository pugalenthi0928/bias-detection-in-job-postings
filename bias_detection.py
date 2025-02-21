import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample biased and unbiased job descriptions
data = {
    "text": [
        "We are looking for a strong and aggressive salesman to drive results.",
        "The ideal candidate is a hardworking and disciplined individual.",
        "We are seeking a dynamic professional with excellent communication skills.",
        "Looking for a compassionate and nurturing female candidate for this role.",
        "A young and energetic team player is preferred.",
    ],
    "label": ["biased", "unbiased", "unbiased", "biased", "biased"]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# NLP Preprocessing Function
def preprocess_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

df["processed_text"] = df["text"].apply(preprocess_text)

# Machine Learning Pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

# Train the model
pipeline.fit(df["processed_text"], df["label"])

# Test on new job descriptions
test_samples = [
    "We need a young and driven individual to join our startup.",
    "An experienced and detail-oriented professional is required."
]

processed_samples = [preprocess_text(text) for text in test_samples]

# Predictions
predictions = pipeline.predict(processed_samples)

# Output Results
for i, text in enumerate(test_samples):
    print(f"Job Description: {text}\nPrediction: {predictions[i]}\n")
