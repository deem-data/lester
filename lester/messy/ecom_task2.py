# Based on https://github.com/LittleDevilBig/Systems-for-AI-Quality/blob/main/main.py

__ORIGINAL_CODE_FULL = '''
import pandas as pd
import collections
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.decomposition import TruncatedSVD


def load_data():
    data_product = pd.concat([pd.read_csv('dataset/products-data-{}.tsv'.format(x), sep='\t',
                                          names=['product_id', 'product_category', 'product_title']) for x in range(4)])
    data_reviews = pd.concat([pd.read_csv('dataset/reviews-{}.tsv'.format(x), sep='\t',
                                          names=['product_id', 'product_rating', 'product_review']) for x in [0, 1, 3]])
    data_reviews = pd.concat([data_reviews, pd.read_csv('dataset/reviews-2.tsv', sep='\t',
                                                        names=['product_rating', 'product_id', 'product_review'])])
    data_total = pd.merge(data_product, data_reviews, how='left')
    return data_total


def build_train_test(data_total):
    train_dataset = data_total.sample(frac=0.8, random_state=0)
    test_dataset = data_total.drop(train_dataset.index)
    return train_dataset, test_dataset


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


def train(train_dataset):
    # train model
    print('training model...')
    feature, label = preprocess(train_dataset)
    model = LogisticRegression()
    model.fit(feature, label)
    return model


def test(model, test_dataset):
    # test model
    print('\ntesting model...')
    feature, label = preprocess(test_dataset)
    predict = model.predict(feature)
    print('\nResult: ')
    print('prediction: {}'.format(collections.Counter(predict)))
    print('label: {}'.format(collections.Counter(label)))
    print('accuracy: {}'.format(model.score(feature, label)))
    print('classification report: \n', classification_report(label, predict))
    print('roc_auc_score: {}'.format(roc_auc_score(label, predict)))
    print('confusion matrix: \n', pd.crosstab(label, predict, rownames=['True'], colnames=['Predicted'], margins=True))
    print('done.')


def main():
    dataset = load_data()
    train_dataset, test_dataset = build_train_test(dataset)
    model = train(train_dataset)
    test(model, test_dataset)


if __name__ == '__main__':
    main()
'''


def __datasplit__original(data_total):
    train_dataset = data_total.sample(frac=0.8, random_state=0)
    test_dataset = data_total.drop(train_dataset.index)
    return train_dataset, test_dataset


def __featurise_original(data_total, lower_dimension=100):
    import re
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    # build vocabulary
    vocab = [re.sub('[^A-Za-z]+', ' ', str(title)).strip().lower()
             + re.sub('[^A-Za-z]+', ' ', str(comment)).strip().lower()
             for title, comment in zip(data_total['product_name'], data_total['review'])]
    vec = TfidfVectorizer()
    feature = vec.fit_transform(vocab)
    # reduce feature dimension from vocab size to lower_dimension using SVD
    svd = TruncatedSVD(n_components=lower_dimension, n_iter=7, random_state=42)
    feature = svd.fit_transform(feature)
    rating = data_total['rating']
    # add rating to the first column of feature
    feature = np.insert(feature, 0, rating, axis=1)
    return feature


def __datasplit__manual(tracked_dataframe):
    import lester as ld
    train, test = ld.split(tracked_dataframe, 0.8)
    return train, test


def __featurise_manual(data, lower_dimension=100):
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

    return featuriser.fit_transform(data)
