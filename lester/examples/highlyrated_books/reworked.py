from lester.context import (datasource, prepare, split, encode_features, encode_target, train_model, DataframeDialect,
                            EstimatorTransformerDialect)
import os
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, LabelEncoder
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from skorch import NeuralNetBinaryClassifier
from skorch.dataset import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@prepare(
    dialect=DataframeDialect.POLARS,
    sources=[
        datasource('books', track_provenance_by=['goodreads_book_id']),
        datasource('categories', track_provenance_by=['tag_id']),
        datasource('book_tags')])
def label_books(books, categories, book_tags):

    english_books = books \
        .drop_nulls() \
        .filter(pl.col("language_code") == 'eng')

    popular_categories = categories.filter(pl.col("popularity") >= 10)
    categories_with_books = popular_categories.join(book_tags, on='tag_id')

    labeled_books = english_books.join(categories_with_books, on='goodreads_book_id')
    labeled_books = labeled_books.with_column((pl.col("average_rating") > 4.2).alias("is_highly_rated"))

    return labeled_books


@split(dialect=DataframeDialect.PANDAS)
def random_split(data, random_seed):
    return train_test_split(data, test_size=0.2, random_state=random_seed)


@encode_features(dialect=EstimatorTransformerDialect.SKLEARN)
def encode_books():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(column_slice):
        texts = [' '.join(column) for column in column_slice.values]
        return embedding_model.encode(texts)

    return ColumnTransformer(transformers=[
        ('categorical', OneHotEncoder(handle_unknown='ignore'), ['authors', 'tag_id', 'original_publication_year']),
        ('numerical', StandardScaler(), ['work_text_reviews_count']),
        ('embeddings', FunctionTransformer(embed), ['title']),
    ], sparse_threshold=0.0)


@encode_target(target_column='is_highly_rated')
def encode_target():
    return LabelEncoder()


class MLP(nn.Module):

    def __init__(self, num_features, hidden_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.layers(x))


class TorchDataset(Dataset):
    def transform(self, X, y):
        X_transformed = X.astype(np.float32)

        if y is not None:
            y_transformed = y.astype(np.float32)
        else:
            y_transformed = None

        return super().transform(X_transformed, y_transformed)


@train_model(dialect=EstimatorTransformerDialect.SKLEARN)
def neural_network(num_features):
    return NeuralNetBinaryClassifier(
        MLP(num_features=num_features, hidden_size=256),
        max_epochs=50,
        lr=0.001,
        iterator_train__shuffle=True,
        criterion=torch.nn.BCELoss,
        optimizer=torch.optim.Adam,
        dataset=TorchDataset
    )
