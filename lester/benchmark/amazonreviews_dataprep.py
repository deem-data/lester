# Based on https://github.com/aayush210789/Deception-Detection-on-Amazon-reviews-dataset/blob/master/SVM_model.ipynb
from lester.benchmark import DataprepCodeTransformationTask


class AmazonreviewsDataprepTask(DataprepCodeTransformationTask):

    @property
    def original_code(self):
        return """
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

# load data from a file and append it to the rawData
def loadData(path, Text=None):
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for line in reader:
            (Id, Text, Label) = parseReview(line)
            rawData.append((Id, Text, Label))
            preprocessedData.append((Id, preProcess(Text), Label))

# Convert line from input file into an id/text/label tuple
def parseReview(reviewLine):
    # Should return a triple of an integer, a string containing the review, and a string indicating the label
    s=""
    if reviewLine[1]=="__label1__":
        s = "fake"
    else: 
        s = "real"
    return (reviewLine[0], reviewLine[8], s)              
    
# loading reviews
rawData = []          # the filtered data from the dataset file (should be 21000 samples)
preprocessedData = [] # the preprocessed reviews (just to see how your preprocessing is doing)
trainData = []        # the training data as a percentage of the total dataset (currently 80%, or 16800 samples)
testData = []         # the test data as a percentage of the total dataset (currently 20%, or 4200 samples)

# the output classes
fakeLabel = 'fake'
realLabel = 'real'

rawData = loadData(reviewPath)           
"""

    def input_arg_names(self):
        return ['reviewPath']

    def input_schemas(self):
        return [['DOC_ID', 'LABEL', 'RATING', 'VERIFIED_PURCHASE', 'PRODUCT_CATEGORY', 'PRODUCT_ID',
                'PRODUCT_TITLE', 'REVIEW_TITLE', 'REVIEW_TEXT']]

    def output_columns(self):
        return ['id', 'rating', 'verified_purchase', 'product_category', 'text']

    def run_manually_rewritten_code(self, params):
        import lester as ld
        reviews = ld.read_csv(params['reviewPath'], header=0, sep='\t')

        reviews = reviews.project('label', ['LABEL'], lambda label: 'fake' if label == "__label1__" else 'real')
        reviews = reviews[['DOC_ID', 'RATING',	'VERIFIED_PURCHASE', 'PRODUCT_CATEGORY', 'REVIEW_TEXT', 'label']]
        reviews = reviews.rename({'DOC_ID': 'id', 'RATING': 'rating', 'VERIFIED_PURCHASE': 'verified_purchase',
                                  'PRODUCT_CATEGORY': 'product_category', 'REVIEW_TEXT': 'text'})
        return reviews

    def evaluate_transformed_code(self, transformed_code):

        params = {'reviewPath': './data/amazon_reviews_small.txt'}

        variables_for_exec = {}
        exec(transformed_code, variables_for_exec)

        generated_result = variables_for_exec['__dataprep'](reviewPath=params['reviewPath'])

        manual_result = self.run_manually_rewritten_code(params)

        data_diff = manual_result.df[self.output_columns()].compare(generated_result.df[self.output_columns()])
        assert len(data_diff) == 0
