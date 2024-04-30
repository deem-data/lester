from lester.context import (split, encode_features, encode_target, train_model, DataframeDialect,
                            EstimatorTransformerDialect)

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, LabelEncoder
from sentence_transformers import SentenceTransformer

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
    ])


@encode_target(target_column='is_highly_rated')
def encode_target():
    return LabelEncoder()
