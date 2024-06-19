# --- GENERATED CODE [START] ---
import lester as lt
from transformers import pipeline
from dateutil import parser

target_countries = ['UK', 'DE', 'FR']
sentiment_predictor = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')


def matches_usecase(text):
    return "complaint" in text or "bank account" in text


def sanitize(text):
    return text.lower()


def _lester_dataprep(customers_path, mails_path):
    # Load customer data
    customer_df = lt.read_csv(customers_path, header=None,
                              names=['customer_id', 'customer_email', 'bank', 'country', 'level'])
    customer_df = customer_df.filter('country in @target_countries')
    customer_df = customer_df.project(target_column='is_premium', source_columns=['level'],
                                      func=lambda row: row['level'] == 'premium')

    # Select relevant columns for merging
    customer_df = customer_df[['customer_email', 'bank', 'country', 'is_premium']]

    # Load mails data
    mails_df = lt.read_csv(mails_path, header=None, names=['mail_id', 'email', 'raw_date', 'mail_subject', 'mail_text'])
    mails_df = mails_df.project(target_column='mail_date', source_columns=['raw_date'],
                                func=lambda row: parser.parse(row['raw_date']))
    mails_df = mails_df.filter('mail_date.dt.year >= 2022')

    mails_df = mails_df.filter('mail_text.apply(@matches_usecase)')

    # Merge dataframes
    merged_df = lt.join(mails_df, customer_df, left_on='email', right_on='customer_email')

    # Process and assign new columns
    merged_df = merged_df.project(target_column='title', source_columns=['mail_subject'],
                                  func=lambda row: sanitize(row['mail_subject']))
    merged_df = merged_df.project(target_column='text', source_columns=['mail_text'],
                                  func=lambda row: sanitize(row['mail_text']))
    merged_df = merged_df.project(target_column='sentiment', source_columns=['mail_text'],
                                  func=lambda row: sentiment_predictor(row['mail_text'])[0]['label'].lower())

    # Select the required columns
    result_df = merged_df[['title', 'text', 'bank', 'country', 'sentiment', 'is_premium']]

    return result_df
# --- GENERATED CODE [END] ---

# --- GENERATED CODE [START] ---
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sentence_transformers import SentenceTransformer


class SentenceEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = None

    def fit(self, X, y=None):
        self.model = SentenceTransformer(self.model_name)
        return self

    def transform(self, X, y=None):
        X = [elem[0] for elem in X.values]  # NEEDED TO BE MANUALLY ADDED!
        return self.model.encode(X)


class WordCountTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = [elem[0] for elem in X.values]  # NEEDED TO BE MANUALLY ADDED!
        return np.array([len(text.split(" ")) for text in X]).reshape(-1, 1)


def encode_features():
    subject_pipeline = Pipeline([
        ('embedding', SentenceEmbeddingTransformer()),
    ])

    text_pipeline = Pipeline([
        ('embedding', SentenceEmbeddingTransformer()),
    ])

    subject_length_pipeline = Pipeline([
        ('length', WordCountTransformer()),
        ('scaler', StandardScaler()),
    ])

    country_pipeline = Pipeline([
        ('onehot', OneHotEncoder(sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('subject_embedding', subject_pipeline, ['title']),
        ('text_embedding', text_pipeline, ['text']),
        ('subject_length', subject_length_pipeline, ['title']),
        ('country', country_pipeline, ['country'])
    ])

    return preprocessor
# --- GENERATED CODE [END] ---


# --- GENERATED CODE [START] ---
def extract_label(df: pd.DataFrame) -> np.ndarray:
    label = np.where((df['sentiment'] == 'negative') & (df['is_premium'] == True), 1.0, 0.0)
    return label
# --- GENERATED CODE [END] ---
