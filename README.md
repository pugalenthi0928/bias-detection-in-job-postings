# Bias Detection in Job Postings

This project detects potential bias in job descriptions using NLP techniques. It leverages **spaCy** and **Scikit-learn** to analyze text and classify job postings as "biased" or "unbiased."

## Features
- Preprocesses job descriptions using **spaCy**
- Converts text into numerical features using **TF-IDF**
- Classifies descriptions with **Naive Bayes**
- Provides predictions on new job descriptions

## Installation
```bash
pip install spacy scikit-learn pandas
python -m spacy download en_core_web_sm
