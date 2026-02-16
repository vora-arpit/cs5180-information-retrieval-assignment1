#-------------------------------------------------------------
# AUTHOR: Arpit Vora
# FILENAME: search_engine.py
# SPECIFICATION:
# Program computes document ranking for query "I love dogs"
# using binary unigram + bigram representation with:
# tokenization, normalization, stopword removal, and stemming.
# FOR: CS 5180 - Assignment #1
# TIME SPENT: 10 hours
#-----------------------------------------------------------*/

# ---------------------------------------------------------
# Importing some Python libraries
# ---------------------------------------------------------
import csv
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer

documents = []

# ---------------------------------------------------------
# Reading the data in a csv file
# ---------------------------------------------------------
with open('collection.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            documents.append(row[0])

# ---------------------------------------------------------
# Print original documents
# ---------------------------------------------------------
print("Original Documents:")
print(documents)
print()

# ---------------------------------------------------------
# Custom tokenizer with normalization + stemming
# ---------------------------------------------------------
stemmer = PorterStemmer()

def stem_tokenizer(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return [stemmer.stem(token) for token in tokens]

# Stop words
stop_words = [
    "i", "she", "her", "they", "their",
    "a", "an", "and", "the"
]

# ---------------------------------------------------------
# Create vectorizer
# ---------------------------------------------------------
vectorizer = CountVectorizer(
    analyzer='word',
    tokenizer=stem_tokenizer,
    token_pattern=None,
    stop_words=stop_words,
    ngram_range=(1, 2),
    binary=True
)

# ---------------------------------------------------------
# Fit vectorizer to documents
# ---------------------------------------------------------
vectorizer.fit(documents)
document_matrix = vectorizer.transform(documents)

# ---------------------------------------------------------
# Show vocabulary
# ---------------------------------------------------------
vocab = vectorizer.get_feature_names_out().tolist()
print("Vocabulary:", vocab)
print()

# ---------------------------------------------------------
# Encode query
# ---------------------------------------------------------
query = ["I love dogs"]
query_vector = vectorizer.transform(query)

# ---------------------------------------------------------
# Convert to Python lists
# ---------------------------------------------------------
doc_vectors = document_matrix.toarray().tolist()
query_vector = query_vector.toarray().tolist()[0]
print("Document Vectors:")
for i, vec in enumerate(doc_vectors):
    print(f"d{i+1}:", vec)

print("\nQuery Vector:", query_vector)
print()

# ---------------------------------------------------------
# Compute dot product
# ---------------------------------------------------------
scores = []
for doc_vector in doc_vectors:
    score = sum(q * d for q, d in zip(query_vector, doc_vector))
    scores.append(score)

print("Scores:", scores)

# ---------------------------------------------------------
# Sort documents by score (descending)
# ---------------------------------------------------------
ranking = sorted(
    [(i+1, score) for i, score in enumerate(scores)],
    key=lambda x: x[1],
    reverse=True
)

print("Ranking (DocID, Score):", ranking)
