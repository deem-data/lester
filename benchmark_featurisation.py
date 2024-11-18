from lester.benchmark.creditcard_featurisation import CreditcardFeaturisationTask
from lester.benchmark.ldb_featurisation import LdbFeaturisationTask
from lester.benchmark.titanic_featurisation import TitanicFeaturisationTask

CREDITCARD_FEATURISATION_CODE = """
def __featurise():
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    class SentenceEmbeddingTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, model_name="all-mpnet-base-v2"):
            self.model_name = model_name
            self.model = SentenceTransformer(self.model_name)

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return self.model.encode(X)

    class TextLengthTransformer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            lengths = np.array([len(text.split(" ")) for text in X]).reshape(-1, 1)
            return lengths

    sentence_embedder = SentenceEmbeddingTransformer()
    country_indices = {'DE': 0, 'FR': 1, 'UK': 2}
    country_encoder = OneHotEncoder(categories=[list(country_indices.keys())])

    column_transformer = ColumnTransformer(
        transformers=[
            ("title_embedding", sentence_embedder, "title"),
            ("text_embedding", sentence_embedder, "text"),
            ("title_length", Pipeline([
                ("length", TextLengthTransformer()),
                ("scaler", StandardScaler())
            ]), "title"),
            ("country_onehot", country_encoder, ["country"])
        ]
    )

    return column_transformer
"""

LDB_FEATURISATION_CODE = """
def __featurise():
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer
    import re

    # Function to preprocess text data
    def preprocess_text(data_total):
        vocab = [re.sub('[^A-Za-z]+', ' ', str(title)).strip().lower() +
                 re.sub('[^A-Za-z]+', ' ', str(comment)).strip().lower()
                 for title, comment in zip(data_total['product_name'], data_total['review'])]
        return vocab

    # Function to extract ratings
    def extract_rating(rating_series):
        return rating_series.values.reshape(-1, 1)

    # Create a pipeline for text processing and dimensionality reduction
    text_pipeline = Pipeline([
        ('text_preprocessing', FunctionTransformer(preprocess_text, validate=False)),
        ('tfidf', TfidfVectorizer()),
        ('svd', TruncatedSVD(n_components=100, n_iter=7, random_state=42))
    ])

    # Create a pipeline for processing numeric rating
    rating_pipeline = Pipeline([
        ('extract_rating', FunctionTransformer(extract_rating, validate=False))
    ])

    # Combine all features using ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('text_features', text_pipeline, ['product_name', 'review']),
        ('rating_feature', rating_pipeline, ['rating'])
    ])

    return preprocessor
"""

TITANIC_FEATURISATION_CODE = """
def __featurise():
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import FunctionTransformer

    # Define transformers
    numerical_features = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_features = ['Sex', 'Embarked']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor
"""

print('CreditcardFeaturisation...')
#creditcard_task = CreditcardFeaturisationTask()
#creditcard_task.evaluate_transformed_code(CREDITCARD_FEATURISATION_CODE)

print('LdbFeaturisation...')
ldb_task = LdbFeaturisationTask()
ldb_task.evaluate_transformed_code(LDB_FEATURISATION_CODE)

print('TitanicFeaturisation...')
titanic_task = TitanicFeaturisationTask()
titanic_task.evaluate_transformed_code(TITANIC_FEATURISATION_CODE)
