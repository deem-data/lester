from lester.context import split, encode_features, DataframeDialect, EstimatorTransformerDialect

from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, Tokenizer, StringIndexer, OneHotEncoder, StandardScaler, VectorAssembler


@split(dialect=DataframeDialect.PYSPARK)
def random_split(data, random_seed):
    return data.randomSplit([0.8, 0.2], seed=random_seed)


@encode_features(dialect=EstimatorTransformerDialect.SPARKML)
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
