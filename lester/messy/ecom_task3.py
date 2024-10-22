# Based on https://github.com/azarijafari/E-commerce-Product-Classification/blob/main/Product%20Classification.ipynb

import pandas as pd
import lester as ld
import numpy as np


def __featurise_original(data):
    import re
    import string
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from gensim.models import FastText
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import Tokenizer

    nltk.download('wordnet')  # Download the wordnet resources - semantic information and similarity of words
    nltk.download('punkt')  # Download the punkt resources for tokenization

    def text_lowercase(text):
        return text.lower()

    # Cleaning the text from emails, symbols, and web links
    def remove_urls(text):
        new_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
        return new_text

    def remove_numbers(text):
        result = re.sub(r'\d+', '', text)
        return result

    # Cleaning punctuation marks from the text - retaining only letters and numbers.
    def remove_punctuation(text):
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)

    def tokenize(text):
        text = word_tokenize(text)
        return text

    lemmatizer = WordNetLemmatizer()
    def lemmatize(text):
        text = [lemmatizer.lemmatize(token) for token in text]
        return text

    def preprocessing(text):
        text = text_lowercase(text)
        text = remove_urls(text)
        text = remove_numbers(text)
        text = remove_punctuation(text)
        text = tokenize(text)
        text = lemmatize(text)
        text = ' '.join(text)
        return text


    count = 0
    documents = []

    for entry in data['product_name']:
        count += 1
        documents.append(preprocessing(str(entry)))

    data['product_name'] = documents


    # Review_text cleaning
    count = 0
    documents1 = []

    for entry in data['review']:
        count += 1
        documents1.append(preprocessing(str(entry)))

    data['review'] = documents1


    data["rating"].loc[data["rating"]==1]=-1.0
    data["rating"].loc[data["rating"]==2]=-0.5
    data["rating"].loc[data["rating"]==3]=0.0
    data["rating"].loc[data["rating"]==4]=0.5
    data["rating"].loc[data["rating"]==5]=1.0


    # Text Tokenization
    max_words = 10000  # Maximum number of allowed words

    tokenizer = Tokenizer(num_words=max_words, oov_token="")
    tokenizer.fit_on_texts(data['product_name'] + ' ' + data['review'])

    # Convert text to numerical vectors
    X_text_sequences = tokenizer.texts_to_sequences(data['product_name'] + ' ' + data['review'])


    # Train FastText model
    sentences = [text.split() for text in data['product_name'] + ' ' + data['review']]
    fasttext_model = FastText(sentences, vector_size=64, window=5, min_count=1, workers=4)

    # Create an Embedding matrix
    embedding_matrix = np.zeros((max_words, 64))
    for word, i in tokenizer.word_index.items():
        if i < max_words:
            if word in fasttext_model.wv:
                embedding_matrix[i] = fasttext_model.wv[word]

    # Convert data to numerical features using "Bag of Words" Embedding method
    X_embedding = np.array([np.mean([embedding_matrix[i] for i in text], axis=0) for text in X_text_sequences])

    # Concatenate Embedding features with ratings
    X_embedding = np.concatenate([X_embedding, data['rating'].values.reshape(-1, 1)], axis=1)
    return X_embedding


def __featurise_manual(data):
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.exceptions import NotFittedError
    import numpy as np
    from gensim.models import FastText
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.pipeline import Pipeline
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import Tokenizer

    def concat_and_clean(df):
        import re
        import string
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        translator = str.maketrans('', '', string.punctuation)

        text = ' '.join(df)
        text = text.lower()
        text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", text).split())
        text = re.sub(r'\d+', '', text)
        text = text.translate(translator)
        text = word_tokenize(text)
        text = [lemmatizer.lemmatize(token) for token in text]
        text = ' '.join(text)
        return text


    class TextEmbeddings(BaseEstimator, TransformerMixin):

        def __init__(self, max_words=10000):
            self.max_words = max_words
            self.is_fitted_ = False
            self.tokenizer_ = None
            self.embedding_matrix_ = None

        def fit(self, X, y=None):

            self.tokenizer_ = Tokenizer(num_words=self.max_words, oov_token="")
            self.tokenizer_.fit_on_texts(X)

            sentences = [text.split() for text in X]
            fasttext_model = FastText(sentences, vector_size=64, window=5, min_count=1, workers=4)

            self.embedding_matrix_ = np.zeros((self.max_words, 64))
            for word, i in self.tokenizer_.word_index.items():
                if i < self.max_words:
                    if word in fasttext_model.wv:
                        self.embedding_matrix_[i] = fasttext_model.wv[word]

            self.is_fitted_ = True
            return self

        def transform(self, X):
            if not self.is_fitted_:
                raise NotFittedError

            X_text_sequences = self.tokenizer_.texts_to_sequences(list(X))
            X_embedding = np.array([np.mean([self.embedding_matrix_[i] for i in text], axis=0) for text in X_text_sequences])

            return X_embedding

    def normalise_rating(r):
        return (r / 2.0) - 1.5

    text_transformation = Pipeline([
        ('clean', FunctionTransformer(lambda df: df.apply(concat_and_clean, axis=1))),
        ('embed', TextEmbeddings()),
    ])

    featuriser = ColumnTransformer(transformers=[
        ('rating', FunctionTransformer(lambda df: df.apply(normalise_rating, axis=1)), ['rating']),
        ('textual_features', text_transformation, ['product_name', 'review']),
    ])

    return featuriser.fit_transform(data)
