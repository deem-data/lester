from lester.benchmark import FeaturisationCodeTransformationTask


class CreditcardFeaturisationTask(FeaturisationCodeTransformationTask):

    @property
    def original_code(self):
        return '''
import numpy as np
from sentence_transformers import SentenceTransformer

sentence_embedder = SentenceTransformer("all-mpnet-base-v2")
country_indices = {'DE': 0, 'FR': 1, 'UK': 2}

titles = []
texts = []
countries = []
        
with open(".scratchspace/__intermediate.csv") as file:
    for line in file:
        parts = line.strip().split("\t")
        title, text, bank, country, sentiment, is_premium = parts

        titles.append(title)
        texts.append(text)
        countries.append(country)

subject_embeddings = sentence_embedder.encode(titles)
text_embeddings = sentence_embedder.encode(texts)
title_lengths = [len(title.split(" ") for titles in titles]
title_lengths = (title_lengths - np.mean(title_lengths)) / np.std(title_lengths)

country_onehot = np.zeros((len(countries), len(country_indices)))
for row, country in enumerate(countries):
    country_onehot[row, country_indices[country]] = 1.0


X = np.concatenate((
    subject_embeddings,
    text_embeddings,
    title_lengths.reshape(-1,1),
    country_onehot
), axis=1)        
'''

    @property
    def input_schema(self):
        return ['title', 'text', 'bank', 'country', 'sentiment', 'is_premium']

    def run_manually_rewritten_code(self, params):

        import numpy as np
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
                X = [elem[0] for elem in X.values]
                return self.model.encode(X)

        class WordCountTransformer(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self

            def transform(self, X, y=None):
                X = [elem[0] for elem in X.values]
                return np.array([len(text.split(" ")) for text in X]).reshape(-1, 1)

        subject_length_pipeline = Pipeline([('length', WordCountTransformer()), ('scaler', StandardScaler())])
        country_pipeline = Pipeline([('onehot', OneHotEncoder(sparse_output=False))])

        preprocessor = ColumnTransformer([
            ('subject_embedding', SentenceEmbeddingTransformer(), ['title']),
            ('text_embedding', SentenceEmbeddingTransformer(), ['text']),
            ('subject_length', subject_length_pipeline, ['title']),
            ('country', country_pipeline, ['country'])
        ])

        return preprocessor

    def evaluate_transformed_code(self, transformed_code):
        variables_for_exec = {}
        exec(transformed_code, variables_for_exec)

        import pandas as pd
        prepared_data = pd.read_csv("data/benchmark/creditcard_featurisation.csv")

        generated_column_transformer = variables_for_exec['__featurise']()
        generated_X = generated_column_transformer.fit_transform(prepared_data)

        manual_column_transformer = self.run_manually_rewritten_code({})
        manual_X = manual_column_transformer.fit_transform(prepared_data)

        assert generated_X.shape == manual_X.shape


