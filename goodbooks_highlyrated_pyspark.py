from lester.context import datasource, prepare, split, encode_features, model_training
from lester.pyspark.runner import run

from pyspark.ml import Pipeline
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import when, col
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer, StringIndexer, OneHotEncoder, StandardScaler, VectorAssembler


@prepare(
    books=datasource('books', track_provenance_by=['goodreads_book_id']),
    categories=datasource('categories', track_provenance_by=['tag_id']),
    book_tags=datasource('book_tags'))
def label_books(books, categories, book_tags):

    english_books = books \
        .dropna() \
        .filter("language_code == 'eng'")

    popular_categories = categories.filter("popularity >= 10")
    categories_with_books = popular_categories.join(book_tags, on='tag_id')

    labeled_books = english_books.join(categories_with_books, on='goodreads_book_id')
    labeled_books = labeled_books.withColumn('label', when(col('average_rating') > 4.2, 1.0).otherwise(0.0))
    labeled_books = labeled_books \
        .withColumn('work_text_reviews_count', col('work_text_reviews_count').cast(IntegerType()))

    return labeled_books


@split()
def random_split(data, random_seed):
    return data.randomSplit([0.8, 0.2], seed=random_seed)


@encode_features()
def encode_books():
    stages = []

    # One-hot encode categorical features
    categorical_columns = ['authors', 'tag_id', 'original_publication_year']
    for column in categorical_columns:
        stages.append(StringIndexer(inputCol=column, outputCol=f"{column}Index", handleInvalid='keep'))

    categorical_indexes = [f'{column}Index' for column in categorical_columns]
    categorical_features = [f'{column}Vec' for column in categorical_columns]
    encoder = OneHotEncoder(inputCols=categorical_indexes, outputCols=categorical_features)
    stages.append(encoder)

    numerical_columns = ['work_text_reviews_count']

    numerical_assembler = VectorAssembler(inputCols=numerical_columns, outputCol='numerical_features_raw')
    stages.append(numerical_assembler)
    # Normalize numerical features
    scaler = StandardScaler(inputCol='numerical_features_raw', outputCol='numerical_features', withMean=True)
    stages.append(scaler)

    # Hash word features
    tokenizer = Tokenizer(inputCol="title", outputCol="words")
    hashing = HashingTF(inputCol="words", numFeatures=100, outputCol="text_features")

    stages.append(tokenizer)
    stages.append(hashing)

    # Concatenate all features
    assembler = VectorAssembler(inputCols=categorical_features + ['numerical_features', 'text_features'],
                                outputCol="features")
    stages.append(assembler)

    return Pipeline(stages=stages)


@model_training()
def logreg_with_fixed_hyperparams():
    return LogisticRegression(maxIter=10, regParam=0.001)


run()
