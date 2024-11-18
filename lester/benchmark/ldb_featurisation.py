# https://github.com/LittleDevilBig/Systems-for-AI-Quality/blob/main/main.py
from lester.benchmark import FeaturisationCodeTransformationTask


class LdbFeaturisationTask(FeaturisationCodeTransformationTask):

    @property
    def original_code(self):
        return """
def preprocess(data_total, lower_dimension=100):
    # build label set
    label = data_total['product_category'] == 'Jewelry'

    # build vocabulary
    vocab = [re.sub('[^A-Za-z]+', ' ', str(title)).strip().lower()
             + re.sub('[^A-Za-z]+', ' ', str(comment)).strip().lower()
             for title, comment in zip(data_total['product_title'], data_total['product_review'])]
    vec = TfidfVectorizer()
    feature = vec.fit_transform(vocab)
    print('feature shape: {}'.format(feature.shape))
    # reduce feature dimension from vocab size to lower_dimension using SVD
    # this technique also made the dimension of training data and test data consistent
    svd = TruncatedSVD(n_components=lower_dimension, n_iter=7, random_state=42)
    feature = svd.fit_transform(feature)
    print('feature shape after dimension reduction: {}'.format(feature.shape))
    rating = data_total['product_rating']
    # add rating to the first column of feature
    feature = np.insert(feature, 0, rating, axis=1)
    print('feature shape after adding rating: {}'.format(feature.shape))
    return feature, label
"""

    @property
    def input_schema(self):
        return ['product_id', 'product_category', 'product_name', 'rating', 'review']

    def run_manually_rewritten_code(self, params):
        import re
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import FunctionTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import Pipeline

        def concat_and_clean(df):
            concatenated = ''.join(df)
            cleaned = re.sub('[^A-Za-z]+', ' ', concatenated).strip().lower()
            return cleaned

        lower_dimension = 100

        text_transformation = Pipeline([
            ('clean', FunctionTransformer(lambda df: df.apply(concat_and_clean, axis=1))),
            ('tfidf', TfidfVectorizer()),
            ('svd', TruncatedSVD(n_components=lower_dimension, n_iter=7, random_state=42)),
        ])

        featuriser = ColumnTransformer(transformers=[
            ('rating', FunctionTransformer(lambda x: x), ['rating']),
            ('textual_features', text_transformation, ['product_name', 'review']),
        ])

        return featuriser

    def evaluate_transformed_code(self, transformed_code):
        variables_for_exec = {}
        exec(transformed_code, variables_for_exec)

        import pandas as pd
        prepared_data = pd.read_csv("data/benchmark/ldb_featurisation.csv")

        generated_column_transformer = variables_for_exec['__featurise']()
        generated_X = generated_column_transformer.fit_transform(prepared_data)

        manual_column_transformer = self.run_manually_rewritten_code({})
        manual_X = manual_column_transformer.fit_transform(prepared_data)

        assert generated_X.shape == manual_X.shape
