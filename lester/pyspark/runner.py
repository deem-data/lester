from lester.context import LesterContext
from lester.duckframe import from_tracked_source, from_source
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics
from lester.pyspark.dataframe import PysparkDuckframe
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def __load_lester_datasource(source):

    ctx = LesterContext()

    name_to_path = {
        'ratings': 'data/ratings.csv',
        'books': 'data/books.csv',
        'categories': 'data/categories.csv',
        'book_tags': 'data/book_tags.csv',
        'tags': 'data/tags.csv',
    }

    path = name_to_path[source.name]

    if len(source.track_provenance_by) > 0:
        source_id = ctx.source_counter
        ctx.source_counter += 1
        duckframe = from_tracked_source(source.name, path, source.track_provenance_by, source_id)
    else:
        duckframe = from_source(source.name, path)

    return PysparkDuckframe(duckframe)


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
        datasouce_args[kwarg_name] = __load_lester_datasource(source)

    eprint("Executing relational data preparation")
    prepared_data_as_duckframe = ctx.prepare_function(**datasouce_args)
    prepared_data_as_pandas = prepared_data_as_duckframe.duckframe.relation.to_df()
    prepared_data = spark.createDataFrame(prepared_data_as_pandas)

    print(prepared_data_as_duckframe.duckframe.column_provenance)

    eprint("Splitting prepared data")
    intermediate_train, intermediate_test = ctx.split_function(prepared_data, random_seed)
    # TODO remove the provenance columns here

    eprint("Encoding training data")
    feature_transformer = ctx.encode_features_function()
    model_training = ctx.model_training_function()

    model = Pipeline(stages=[feature_transformer, model_training]).fit(intermediate_train)

    predictions = model.transform(intermediate_test)

    predictions_and_labels = predictions.select(['prediction', 'label']).rdd \
        .map(lambda row: (row['prediction'], float(row['label'])))

    metrics = MulticlassMetrics(predictions_and_labels)
    print(f'Accuracy: {metrics.accuracy}')
