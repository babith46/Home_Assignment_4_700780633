"""CS5760 Natural Language Processing — Part II: all three programs.

Install dependencies once:
    python -m pip install nltk spacy numpy
    python -m spacy download en_core_web_sm

Run:
    python Home_Assignment_4_Three_Programs.py
"""

import numpy as np
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import wordpunct_tokenize


# Q1 — NLP Preprocessing Pipeline
FALLBACK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for",
    "from", "had", "has", "have", "he", "her", "his", "i", "in", "is",
    "it", "its", "of", "on", "or", "our", "she", "that", "the", "their",
    "them", "there", "these", "they", "this", "to", "was", "were",
    "will", "with", "you", "your",
}


def nlp_preprocessing(sentence):
    """Tokenize, remove stopwords/punctuation, and apply Porter stemming."""
    tokens = wordpunct_tokenize(sentence)
    try:
        english_stopwords = set(stopwords.words("english"))
    except LookupError:
        english_stopwords = FALLBACK_STOPWORDS

    filtered_tokens = [
        token for token in tokens
        if token.isalpha() and token.lower() not in english_stopwords
    ]
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(token.lower()) for token in filtered_tokens]

    print("Original Tokens:", tokens)
    print("Tokens Without Stopwords:", filtered_tokens)
    print("Stemmed Words:", stemmed_words)
    return tokens, filtered_tokens, stemmed_words


# Q2 — Named Entity Recognition with spaCy
def named_entity_recognition(sentence):
    """Print named entity text, label, and character positions."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("English spaCy model not installed. Run:")
        print("python -m spacy download en_core_web_sm")
        return []

    entities = list(nlp(sentence).ents)
    for entity in entities:
        print(
            f"Entity: {entity.text} | Label: {entity.label_} | "
            f"Start: {entity.start_char} | End: {entity.end_char}"
        )
    if not entities:
        print("No named entities detected.")
    return entities


# Q3 — Scaled Dot-Product Attention
def scaled_dot_product_attention(Q, K, V):
    """Return attention weights and weighted values using row-wise softmax."""
    d_k = K.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    shifted_scores = scores - np.max(scores, axis=-1, keepdims=True)
    exponentials = np.exp(shifted_scores)
    weights = exponentials / np.sum(exponentials, axis=-1, keepdims=True)
    output = weights @ V
    return weights, output


def main():
    print("\n========== PROGRAM 1: NLP PREPROCESSING ==========")
    sentence_1 = "NLP techniques are used in virtual assistants like Alexa and Siri."
    nlp_preprocessing(sentence_1)

    print("\n========== PROGRAM 2: NAMED ENTITY RECOGNITION ==========")
    sentence_2 = (
        " LeBron James as the NBA player of the United States "
        "in Laker and won the NBA Final MVP in 2020."
    )
    named_entity_recognition(sentence_2)

    print("\n========== PROGRAM 3: SCALED DOT-PRODUCT ATTENTION ==========")
    Q = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
    K = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
    V = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    weights, result = scaled_dot_product_attention(Q, K, V)
    print("Attention Weights Matrix:\n", np.round(weights, 3))
    print("Final Output Matrix:\n", np.round(result, 3))


if __name__ == "__main__":
    main()
