import os
import warnings
import uuid

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "False"

# # --- GENERATED CODE [START] ---
# import lester as lt
# from transformers import pipeline
# from dateutil import parser
#
# target_countries = ['UK', 'DE', 'FR']
# sentiment_predictor = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
#
#
# def matches_usecase(text):
#     return "complaint" in text or "bank account" in text
#
#
# def sanitize(text):
#     return text.lower()
#
#
# def _lester_dataprep(customers_path, mails_path):
#     # Load customer data
#     customer_df = lt.read_csv(customers_path, header=None,
#                               names=['customer_id', 'customer_email', 'bank', 'country', 'level'])
#     customer_df = customer_df.filter('country in @target_countries')
#     customer_df = customer_df.project(target_column='is_premium', source_columns=['level'],
#                                       func=lambda row: row['level'] == 'premium')
#
#     # Select relevant columns for merging
#     customer_df = customer_df[['customer_email', 'bank', 'country', 'is_premium']]
#
#     # Load mails data
#     mails_df = lt.read_csv(mails_path, header=None, names=['mail_id', 'email', 'raw_date', 'mail_subject', 'mail_text'])
#     mails_df = mails_df.project(target_column='mail_date', source_columns=['raw_date'],
#                                 func=lambda row: parser.parse(row['raw_date']))
#     mails_df = mails_df.filter('mail_date.dt.year >= 2022')
#
#     mails_df = mails_df.filter('mail_text.apply(@matches_usecase)')
#
#     # Merge dataframes
#     merged_df = lt.join(mails_df, customer_df, left_on='email', right_on='customer_email')
#
#     # Process and assign new columns
#     merged_df = merged_df.project(target_column='title', source_columns=['mail_subject'],
#                                   func=lambda row: sanitize(row['mail_subject']))
#     merged_df = merged_df.project(target_column='text', source_columns=['mail_text'],
#                                   func=lambda row: sanitize(row['mail_text']))
#     merged_df = merged_df.project(target_column='sentiment', source_columns=['mail_text'],
#                                   func=lambda row: sentiment_predictor(row['mail_text'])[0]['label'].lower())
#
#     # Select the required columns
#     result_df = merged_df[['title', 'text', 'bank', 'country', 'sentiment', 'is_premium']]
#
#     return result_df
# # --- GENERATED CODE [END] ---
#
# # --- GENERATED CODE [START] ---
# import numpy as np
# import pandas as pd
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sentence_transformers import SentenceTransformer
#
#
# class SentenceEmbeddingTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, model_name="all-mpnet-base-v2"):
#         self.model_name = model_name
#         self.model = None
#
#     def fit(self, X, y=None):
#         self.model = SentenceTransformer(self.model_name)
#         return self
#
#     def transform(self, X, y=None):
#         X = [elem[0] for elem in X.values]  # NEEDED TO BE MANUALLY ADDED!
#         return self.model.encode(X)
#
#
# class WordCountTransformer(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X, y=None):
#         X = [elem[0] for elem in X.values]  # NEEDED TO BE MANUALLY ADDED!
#         return np.array([len(text.split(" ")) for text in X]).reshape(-1, 1)
#
#
# def encode_features():
#     subject_pipeline = Pipeline([
#         ('embedding', SentenceEmbeddingTransformer()),
#     ])
#
#     text_pipeline = Pipeline([
#         ('embedding', SentenceEmbeddingTransformer()),
#     ])
#
#     subject_length_pipeline = Pipeline([
#         ('length', WordCountTransformer()),
#         ('scaler', StandardScaler()),
#     ])
#
#     country_pipeline = Pipeline([
#         ('onehot', OneHotEncoder(sparse_output=False))
#     ])
#
#     preprocessor = ColumnTransformer([
#         ('subject_embedding', subject_pipeline, ['title']),
#         ('text_embedding', text_pipeline, ['text']),
#         ('subject_length', subject_length_pipeline, ['title']),
#         ('country', country_pipeline, ['country'])
#     ])
#
#     return preprocessor
# # --- GENERATED CODE [END] ---
#
#
# # --- GENERATED CODE [START] ---
# def extract_label(df: pd.DataFrame) -> np.ndarray:
#     label = np.where((df['sentiment'] == 'negative') & (df['is_premium'] == True), 1.0, 0.0)
#     return label
# # --- GENERATED CODE [END] ---

import duckdb
from lester.save_artifacts import _save_as_json, _persist_with_row_provenance, matrix_column_provenance_as_json, _persist_matrices
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def _matrix_column_provenance(column_transformer):
    matrix_column_provenance = []

    for transformer in column_transformer.transformers_:
        name, transformation, columns = transformer
        if name != 'remainder':
            start_index = column_transformer.output_indices_[name].start
            end_index = column_transformer.output_indices_[name].stop
            num_columns_transformed = len(columns)
            ranges = _find_dimensions(transformation, num_columns_transformed, start_index, end_index)

            if ranges is not None:
                matrix_column_provenance += list(zip(columns, ranges))

    merged_column_provenance = {}
    for column, rnge in matrix_column_provenance:
        if column not in merged_column_provenance:
            merged_column_provenance[column] = []
        merged_column_provenance[column].append(rnge)

    return merged_column_provenance


def _find_dimensions(transformation, num_columns_transformed, start_index, end_index):
    if isinstance(transformation, StandardScaler):
        return [slice(start_index + index, start_index + index + 1)
                for index in range(0, num_columns_transformed)]
    elif isinstance(transformation, OneHotEncoder):
        ranges = []
        # TODO We should also include the drop and infrequent features in the calculation
        indices = list([0]) + list(np.cumsum([len(categories) for categories in transformation.categories_]))
        for offset in range(0, len(indices) - 1):
            ranges.append(slice(indices[offset], indices[offset + 1]))
        return ranges
    elif isinstance(transformation, Pipeline):
        _, last_transformation = transformation.steps[-1]
        # TODO We should also look at the intermediate steps of the pipeline in more elaborate cases
        return _find_dimensions(last_transformation, num_columns_transformed, start_index, end_index)
    elif isinstance(transformation, FunctionTransformer):
        # TODO check if the function transformer uses more than one column as input
        return [slice(start_index, end_index)]
    else:
        # Probably a custom transformer, we need to handle it like the function transformer
        return [slice(start_index, end_index)]

import lester as lt
from generated_pipeline_code import _lester_dataprep, encode_features, extract_label

def run_pipeline(name, source_paths, random_seed=42):
    run_id = uuid.uuid4()
    artifact_path = f'.lester/{name}/{run_id}'
    os.makedirs(artifact_path)

    run_id = uuid.uuid4()
    artifact_path = f'.lester/{name}/{run_id}'
    os.makedirs(artifact_path)

    print(f"Starting lester run ({artifact_path})")
    _save_as_json(f'{artifact_path}/source_paths.json', source_paths)

    tracked_df = _lester_dataprep(**source_paths)

    prepared_data = tracked_df.df
    prov_columns = ','.join(tracked_df.row_provenance_columns)
    print("Rows after data preparation:", len(prepared_data))

    intermediate_train, intermediate_test = train_test_split(prepared_data, test_size=0.2, random_state=random_seed)
    _persist_with_row_provenance(intermediate_train, intermediate_test, prov_columns, artifact_path)
    train_df = duckdb.query(f"SELECT * EXCLUDE {prov_columns} FROM intermediate_train").to_df()
    test_df = duckdb.query(f"SELECT * EXCLUDE {prov_columns} FROM intermediate_test").to_df()

    column_provenance = tracked_df.column_provenance
    _save_as_json(f'{artifact_path}/column_provenance.json', column_provenance)

    feature_transformer = encode_features()
    feature_transformer = feature_transformer.fit(train_df)

    matrix_column_provenance = _matrix_column_provenance(feature_transformer)
    _save_as_json(f'{artifact_path}/matrix_column_provenance.json',
                  matrix_column_provenance_as_json(matrix_column_provenance))

    X_train = feature_transformer.transform(train_df)
    y_train = extract_label(train_df)

    num_samples = X_train.shape[0]
    num_features = X_train.shape[1]

    print(f"Training data: {num_samples} samples with {num_features} features")

    import torch
    from neuralnet import custom_mlp

    neuralnet = custom_mlp(num_features)

    model = neuralnet.fit(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float()
    )

    X_test = feature_transformer.transform(test_df)
    y_test = extract_label(test_df)

    y_pred = model.predict(torch.from_numpy(X_test).float())
    print(accuracy_score(y_test, y_pred))

    _persist_matrices(X_train, y_train, X_test, y_test, y_pred, artifact_path)


from generated_pipeline_code import *
lt.make_accessible(locals(), globals())

source_paths = {
    'customers_path': 'data/synthetic_customers_10.csv',
    'mails_path': 'data/synthetic_mails_10.csv'
}

run_pipeline("lester-gen", source_paths)
