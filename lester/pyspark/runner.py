from lester.context import LesterContext
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics
from lester.pyspark.dataframe import TrackedDataframe
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def __load_lester_datasource(ctx, spark, source):

    df = spark.read \
        .option("header", "true") \
        .csv(f'data/{source.name}.csv')

    source_id = None
    if source.track_provenance:
        source_id = ctx.source_counter
        ctx.source_counter += 1

    return TrackedDataframe(df, source_id=source_id)


def run():
    spark = SparkSession.builder\
        .master("local[4]")\
        .config("spark.driver.memory", "8g") \
        .getOrCreate()

    ctx = LesterContext()
    random_seed = 42

    eprint("Loading data from data sources")
    datasouce_args = {}
    for kwarg_name, source in ctx.prepare_function_args.items():
        datasouce_args[kwarg_name] = __load_lester_datasource(ctx, spark, source)

    eprint("Executing relational data preparation")
    prepared_data = ctx.prepare_function(**datasouce_args)

    eprint("Splitting prepared data")
    intermediate_train, intermediate_test = ctx.split_function(prepared_data, random_seed)

    eprint("Encoding training data")
    feature_transformer = ctx.encode_features_function()
    model_training = ctx.model_training_function()

    model = Pipeline(stages=[feature_transformer, model_training]).fit(intermediate_train.df)

    predictions = model.transform(intermediate_test.df)

    predictions_and_labels = predictions.select(['prediction', 'label']).rdd \
        .map(lambda row: (row['prediction'], float(row['label'])))

    metrics = MulticlassMetrics(predictions_and_labels)
    print(f'Accuracy: {metrics.accuracy}')
