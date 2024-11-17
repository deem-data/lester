# Based on https://github.com/aayush210789/Deception-Detection-on-Amazon-reviews-dataset/blob/master/SVM_model.ipynb

import csv                               # csv reader
from sklearn.svm import LinearSVC
from nltk.classify import SklearnClassifier
from random import shuffle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')


from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
import string


def loadData(path, Text=None):
    rawData = []
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for line in reader:
            (Id, Rating, verified_Purchase, product_Category, Text, Label) = parseReview(line)
            rawData.append((Id, Rating, verified_Purchase, product_Category, Text, Label))
    return rawData


def parseReview(reviewLine):

    s=""
    if reviewLine[1]=="__label1__":
        s = "fake"
    else:
        s = "real"
    return (reviewLine[0], reviewLine[2], reviewLine[3],reviewLine[4], reviewLine[8], s)


# MAIN

def __dataprep__original(reviewPath):
    # loading reviews
    rawData = []          # the filtered data from the dataset file (should be 21000 samples)
    preprocessedData = [] # the preprocessed reviews (just to see how your preprocessing is doing)
    #trainData = []        # the training data as a percentage of the total dataset (currently 80%, or 16800 samples)
    #testData = []         # the test data as a percentage of the total dataset (currently 20%, or 4200 samples)

    # the output classes
    fakeLabel = 'fake'
    realLabel = 'real'

    rawData = loadData(reviewPath)

    import pandas as pd
    df = pd.DataFrame.from_records(rawData, columns=['id', 'rating', 'verified_purchase', 'product_category', 'text', 'label'])
    df['id'] = df['id'].astype(int)
    df['rating'] = df['rating'].astype(int)
    return df


def __dataprep__manual(reviewPath):
    import lester as ld
    tracked = ld.read_csv(reviewPath, header=0, sep='\t')

    tracked = tracked.project('label', ['LABEL'], lambda row: 'fake' if row['LABEL']=="__label1__" else 'real')
    tracked = tracked[['DOC_ID', 'RATING',	'VERIFIED_PURCHASE', 'PRODUCT_CATEGORY', 'REVIEW_TEXT', 'label']]
    tracked = tracked.rename({'DOC_ID': 'id', 'RATING': 'rating', 'VERIFIED_PURCHASE': 'verified_purchase',
                              'PRODUCT_CATEGORY': 'product_category', 'REVIEW_TEXT': 'text'})

    return tracked


def splitData(rawData, percentage):
    trainData = []
    testData = []
    dataSamples = len(rawData)
    halfOfData = int(len(rawData)/2)
    trainingSamples = int((percentage*dataSamples)/2)
    for elem in rawData[:trainingSamples] + rawData[halfOfData:halfOfData+trainingSamples]:
        trainData.append(elem)
    for elem in rawData[trainingSamples:halfOfData] + rawData[halfOfData+trainingSamples:]:
        testData.append(elem)

    return trainData, testData


def __datasplit__original(original_result, fraction):
    import pandas as pd
    original_records = list(original_result.to_records(index=False))
    train_records, test_records = splitData(original_records, fraction)

    train = pd.DataFrame.from_records(train_records, columns=['id', 'rating', 'verified_purchase', 'product_category', 'text', 'label'])
    test = pd.DataFrame.from_records(test_records, columns=['id', 'rating', 'verified_purchase', 'product_category', 'text', 'label'])
    return train, test


def __datasplit__manual(tracked, fraction):
    import lester as ld
    train, test = ld.split(tracked, 0.8)
    return train, test


def featurise__original(records):
    import numpy as np
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.tokenize import RegexpTokenizer
    from nltk.stem import WordNetLemmatizer
    from nltk.util import ngrams
    import string

    nltk.download('punkt')
    nltk.download('stopwords')

    # TEXT PREPROCESSING AND FEATURE VECTORIZATION
    # Input: a string of one review
    table = str.maketrans({key: None for key in string.punctuation})
    def preProcess(text):
        # Should return a list of tokens
        lemmatizer = WordNetLemmatizer()
        filtered_tokens=[]
        lemmatized_tokens = []
        stop_words = set(stopwords.words('english'))
        text = text.translate(table)
        for w in text.split(" "):
            if w not in stop_words:
                lemmatized_tokens.append(lemmatizer.lemmatize(w.lower()))
            filtered_tokens = [' '.join(l) for l in nltk.bigrams(lemmatized_tokens)] + lemmatized_tokens
        return filtered_tokens


    featureDict = {} # A global dictionary of features

    def toFeatureVector(Rating, verified_Purchase, product_Category, tokens):
        localDict = {}

        featureDict["R"] = 1
        localDict["R"] = Rating

        featureDict["VP"] = 1

        if verified_Purchase == "N":
            localDict["VP"] = 0
        else:
            localDict["VP"] = 1

        if product_Category not in featureDict:
            featureDict[product_Category] = 1
        else:
            featureDict[product_Category] = +1

        if product_Category not in localDict:
            localDict[product_Category] = 1
        else:
            localDict[product_Category] = +1

        for token in tokens:
            if token not in featureDict:
                featureDict[token] = 1
            else:
                featureDict[token] = +1

            if token not in localDict:
                localDict[token] = 1
            else:
                localDict[token] = +1

        return localDict

    from sklearn.feature_extraction import DictVectorizer
    data = []
    labels = []
    for (_, Rating, verified_Purchase, product_Category, Text, Label) in records:
        data.append(toFeatureVector(Rating, verified_Purchase, product_Category, preProcess(Text)))
        labels.append(Label)

    vectoriser = DictVectorizer(sparse=True)
    features = vectoriser.fit_transform(data)

    return features


def featurise__manual(data):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import FunctionTransformer
    from nltk import word_tokenize
    from nltk.stem import WordNetLemmatizer

    class LemmaTokenizer(object):
        def __init__(self):
            self.lemmatizer = WordNetLemmatizer()

        def __call__(self, texts):
            return [self.lemmatizer.lemmatize(word) for word in word_tokenize(texts)]

    text_transformation = CountVectorizer(tokenizer=LemmaTokenizer(), lowercase=True,
                                          analyzer='word', stop_words='english', ngram_range=(1, 2))

    featuriser = ColumnTransformer(transformers=[
        ('rating', FunctionTransformer(lambda x: x), ['rating']),
        ('categorical', OneHotEncoder(), ['verified_purchase', 'product_category']),
        ('textual_features', text_transformation, 'text'),
    ])

    featuriser.fit_transform(data)
