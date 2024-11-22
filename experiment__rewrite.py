from lester.benchmark.creditcard_dataprep import CreditcardDataprepTask
from lester.benchmark.yichun_dataprep import YichunDataprepTask
from lester.benchmark.amazonreviews_dataprep import AmazonreviewsDataprepTask
from lester.benchmark.creditcard_featurisation import CreditcardFeaturisationTask
from lester.benchmark.ldb_featurisation import LdbFeaturisationTask
from lester.benchmark.titanic_featurisation import TitanicFeaturisationTask
from lester.benchmark.sklearnlogreg_model import SklearnLogisticRegressionTransformationTask
from lester.benchmark.sklearnsvm_model import SklearnSVMTransformationTask
from lester.benchmark.sklearnmlp_model import SklearnMLPTransformationTask


CREDITCARD_DATAPREP_CODE = '''
def __dataprep(customers_file, mails_file):
    import os
    from dateutil import parser
    from transformers import pipeline
    import lester as ld
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    target_countries = ['UK', 'DE', 'FR']
    sentiment_predictor = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

    def sanitize(text):
        return text.lower()

    # Load customer data
    customers_df = ld.read_csv(customers_file, header=None, names=['customer_id', 'customer_email', 'bank', 'country', 'level'], sep=",", parse_dates=False)

    # Filter target countries and create 'is_premium' column
    customers_df = customers_df.filter("country in @target_countries")
    customers_df = customers_df.project('is_premium', ['level'], lambda level: level == 'premium')

    # Select relevant columns
    customers_df = customers_df[['customer_email', 'bank', 'country', 'is_premium']]

    # Load mail data
    mails_df = ld.read_csv(mails_file, header=None, names=['mail_id', 'email', 'raw_date', 'mail_subject', 'mail_text'], sep=",", parse_dates=False)

    # Filter mails from year 2022 onwards
    mails_df = mails_df.project('mail_year', ['raw_date'], lambda raw_date: int(raw_date.split("-")[0]))
    mails_df = mails_df.filter("mail_year >= 2022")

    # Join customer and mail data
    merged_df = ld.join(mails_df, customers_df, left_on='email', right_on='customer_email')

    # Sanitize mail_subject and mail_text
    merged_df = merged_df.project('title', ['mail_subject'], sanitize)
    merged_df = merged_df.project('text', ['mail_text'], sanitize)

    # Predict sentiment
    merged_df = merged_df.project('sentiment', ['mail_text'], lambda mail_text: sentiment_predictor(mail_text)[0]['label'].lower())

    # Select final columns
    result_df = merged_df[['title', 'text', 'bank', 'country', 'sentiment', 'is_premium']]

    return result_df
'''

YICHUN_DATAPREP_CODE = '''
def __dataprep(products_pathes, reviews_pathes):
    import lester as ld

    def read_dataset(products_path, reviews_path, id):
        products_df = ld.read_csv(products_path.format(id), header=None, names=["product_id", "product_category", "product_name"], sep="\t")
        reviews_df = ld.read_csv(reviews_path.format(id), header=None, names=["product_id", "rating", "review"], sep="\t")
        return products_df, reviews_df

    def union_dataset(products_df, reviews_df):
        merged_df = ld.join(products_df, reviews_df, left_on="product_id", right_on="product_id")
        final_df = merged_df[["product_id", "product_category", "product_name", "rating", "review"]]
        return final_df

    products_df_list = []
    reviews_df_list = []
#    for id in range(3):                                                                # MANUALLY REMOVED
#        products_df, reviews_df = read_dataset(products_pathes, reviews_pathes, id)    # MANUALLY REMOVED
    for (products_path, reviews_path) in zip(products_pathes, reviews_pathes):          # MANUALLY ADDED
        products_df, reviews_df = read_dataset(products_path, reviews_path, None)       # MANUALLY ADDED
        products_df_list.append(products_df)
        reviews_df_list.append(reviews_df)

    products_df = ld.union(products_df_list)
    reviews_df = ld.union(reviews_df_list)
    result_df = union_dataset(products_df, reviews_df)

    return result_df
'''

AMAZONREVIEWS_DATAPREP_CODE = '''
def __dataprep(reviewPath):
    import lester as ld
    import numpy as np
    import nltk
    from nltk.tokenize import word_tokenize
    nltk.download('punkt')

    # Convert line from input file into an id/text/label tuple
    def parse_review_label(label):
        return "fake" if label == "__label1__" else "real"

    def pre_process(text):
        # Example preprocessing function (can be customized)
        return " ".join(word_tokenize(text.lower()))

    # Load data into a dataframe
    raw_df = ld.read_csv(reviewPath, header=0, sep='\t', names=['DOC_ID', 'LABEL', 'RATING', 'VERIFIED_PURCHASE',
                                                                'PRODUCT_CATEGORY', 'PRODUCT_ID', 'PRODUCT_TITLE',
                                                                'REVIEW_TITLE', 'REVIEW_TEXT'])

    # Parse the LABEL to real/fake
    raw_df = raw_df.project('parsed_label', ['LABEL'], lambda x: parse_review_label(x[0]))

    # Preprocess the REVIEW_TEXT
    raw_df = raw_df.project('preprocessed_text', ['REVIEW_TEXT'], lambda x: pre_process(x[0]))

    # Select relevant columns and rename them
    result_df = raw_df[['DOC_ID', 'RATING', 'VERIFIED_PURCHASE', 'PRODUCT_CATEGORY', 'REVIEW_TEXT']]
    result_df = result_df.rename({'DOC_ID': 'id', 'RATING': 'rating', 'VERIFIED_PURCHASE': 'verified_purchase',
                                  'PRODUCT_CATEGORY': 'product_category', 'REVIEW_TEXT': 'text'})

    return result_df
'''


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

SKLEARNLOGREG_CODE = """
def __model(num_features):
    import torch
    import torch.nn as nn

    class LogisticRegressionModel(nn.Module):
        def __init__(self, num_features):
            super(LogisticRegressionModel, self).__init__()
            self.linear = nn.Linear(num_features, 1)

        def forward(self, x):
            return torch.sigmoid(self.linear(x))

    model = LogisticRegressionModel(num_features)
    loss = nn.BCELoss()

    return model, loss
"""

SKLEARNSVM_CODE = """
def __model(num_features):
    import torch
    import torch.nn as nn

    class LinearSVC(nn.Module):
        def __init__(self, num_features):
            super(LinearSVC, self).__init__()
            self.linear = nn.Linear(num_features, 1)

        def forward(self, x):
            return self.linear(x)

    model = LinearSVC(num_features)
    loss = nn.HingeEmbeddingLoss()

    return model, loss
"""

SKLEARNMLP_CODE = """
def __model(num_features):
    import torch
    import torch.nn as nn
    
    class CustomModel(nn.Module):
        def __init__(self, input_size):
            super(CustomModel, self).__init__()
            self.hidden1 = nn.Linear(input_size, 64)
            self.hidden2 = nn.Linear(64, 16)
            self.output = nn.Linear(16, 1)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.hidden1(x))
            x = self.relu(self.hidden2(x))
            x = self.output(x)
            return x

    model = CustomModel(num_features)
    loss = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss suitable for binary classification

    return model, loss
"""


print('CreditcardDataprepTask...')
creditcard_task = CreditcardDataprepTask()
creditcard_task.evaluate_transformed_code(CREDITCARD_DATAPREP_CODE)

print('YichunDataprepTask...')
yichun_task = YichunDataprepTask()
yichun_task.evaluate_transformed_code(YICHUN_DATAPREP_CODE)

print('AmazonreviewsDataprepTask...')
amazonreviews_task = AmazonreviewsDataprepTask()
amazonreviews_task.evaluate_transformed_code(AMAZONREVIEWS_DATAPREP_CODE)

print('CreditcardFeaturisation...')
creditcard_task = CreditcardFeaturisationTask()
creditcard_task.evaluate_transformed_code(CREDITCARD_FEATURISATION_CODE)

print('LdbFeaturisation...')
ldb_task = LdbFeaturisationTask()
ldb_task.evaluate_transformed_code(LDB_FEATURISATION_CODE)

print('TitanicFeaturisation...')
titanic_task = TitanicFeaturisationTask()
titanic_task.evaluate_transformed_code(TITANIC_FEATURISATION_CODE)

print("SklearnLogisticRegression...")
logregtask = SklearnLogisticRegressionTransformationTask()
logregtask.evaluate_transformed_code(SKLEARNLOGREG_CODE)

print("SklearnSVM...")
svmtask = SklearnSVMTransformationTask()
svmtask.evaluate_transformed_code(SKLEARNSVM_CODE)

print("SklearnMLP...")
mlptask = SklearnMLPTransformationTask()
mlptask.evaluate_transformed_code(SKLEARNMLP_CODE)
