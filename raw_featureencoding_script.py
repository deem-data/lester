import numpy as np
from sentence_transformers import SentenceTransformer

sentence_embedder = SentenceTransformer("all-mpnet-base-v2")

def count_words(text):
    return len(text.split(" "))

country_indices = {'DE': 0, 'FR': 1, 'UK': 2}

titles = []
title_lengths = []
texts = []
countries = []

with open("./test.csv") as file:
    for line in file:
        parts = line.strip().split("\t")
        title, text, bank, country, sentiment, is_premium = parts

        titles.append(title)
        title_lengths.append(len(title))
        texts.append(text)
        countries.append(country)

subject_embeddings = sentence_embedder.encode(titles)
text_embeddings = sentence_embedder.encode(texts)
title_lengths_column = np.array(title_lengths)
title_lengths_column = (title_lengths_column - np.mean(title_lengths_column)) / np.std(title_lengths_column)

country_onehot = np.zeros((len(countries), len(country_indices)))
for row, country in enumerate(countries):
    country_onehot[row, country_indices[country]] = 1.0


X = np.concatenate((
    subject_embeddings,
    text_embeddings,
    title_lengths_column.reshape(-1,1),
    country_onehot
), axis=1)
