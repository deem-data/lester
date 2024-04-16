from lester.context import datasource, prepare, split, encode_features, encode_target, model_training
from lester.pandas.runner import run

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, LabelEncoder
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@prepare(
    books=datasource('books', track_provenance_by=['goodreads_book_id']),
    categories=datasource('categories', track_provenance_by=['tag_id']),
    book_tags=datasource('book_tags'))
def label_books(books, categories, book_tags):

    english_books = books\
        .dropna()\
        .query("language_code == 'eng'")

    popular_categories = categories.query("popularity >= 10")
    categories_with_books = popular_categories.merge(book_tags, on='tag_id')

    labeled_books = english_books.merge(categories_with_books, on='goodreads_book_id')
    labeled_books['is_highly_rated'] = labeled_books.eval('average_rating > 4.2')

    return labeled_books


@split()
def random_split(data, random_seed):
    return train_test_split(data, test_size=0.2, random_state=random_seed)


@encode_features()
def encode_books():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(column_slice):
        texts = [' '.join(column) for column in column_slice.values]
        return embedding_model.encode(texts)

    return ColumnTransformer(transformers=[
        ('categorical', OneHotEncoder(handle_unknown='ignore'), ['authors', 'tag_id', 'original_publication_year']),
        ('numerical', StandardScaler(), ['work_text_reviews_count']),
        ('embeddings', FunctionTransformer(embed), ['title']),
    ])


@encode_target(target_column='is_highly_rated')
def encode_target():
    return LabelEncoder()


@model_training()
def logreg_with_hpo():
    param_grid = {
        'penalty': ['l2', 'l1'],
        'alpha': [0.0001, 0.001, 0.01],
    }

    learner = SGDClassifier(loss='log_loss', max_iter=1000)
    search = GridSearchCV(learner, param_grid, cv=5, verbose=1, n_jobs=-1)

    return search

run()
