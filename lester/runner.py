import duckdb

from lester.context import LesterContext, DataframeDialect, EstimatorTransformerDialect
from lester.dataframe.duckframe import from_tracked_source, from_source
from lester.dataframe.pandas import PandasDuckframe
from lester.dataframe.pyspark import PysparkDuckframe
from lester.dataframe.polars import PolarsDuckframe


def __load_data(source, name_to_path, dialect):
    path = name_to_path[source.name]
    print(f"  Loading '{source.name}' from '{path}'")

    if len(source.track_provenance_by) > 0:
        duckframe = from_tracked_source(source.name, path, source.track_provenance_by)
    else:
        duckframe = from_source(source.name, path)

    if dialect == DataframeDialect.PANDAS:
        return PandasDuckframe(duckframe)
    elif dialect == DataframeDialect.PYSPARK:
        return PysparkDuckframe(duckframe)
    else:
        return PolarsDuckframe(duckframe)


def run_pipeline(source_paths, random_seed=42):
    ctx = LesterContext()

    # We need an active spark context to use some simple sql functions...
    if ctx.prepare_dialect == DataframeDialect.PYSPARK:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder \
            .master("local[4]") \
            .config("spark.driver.memory", "8g") \
            .getOrCreate()

    print("Accessing data sources.")
    datasource_args = {}
    for source in ctx.prepare_sources:
        datasource_args[source.name] = __load_data(source, source_paths, ctx.prepare_dialect)

    print(f"Executing relational data preparation (with dialect {ctx.prepare_dialect}).")
    prepared_data = ctx.prepare_function(**datasource_args)

    # Materialize into a pandas dataframe
    intermediate = prepared_data.duckframe.relation.to_df()

    # TODO We don't handle more crazy combinations at the moment
    if ctx.model_training_dialect == EstimatorTransformerDialect.SPARKML and \
            ctx.encode_features_dialect == EstimatorTransformerDialect.SPARKML:

        from pyspark.ml import Pipeline
        from pyspark.mllib.evaluation import MulticlassMetrics

        prepared_data = spark.createDataFrame(intermediate)

        print("Splitting prepared data")
        intermediate_train, intermediate_test = ctx.split_function(prepared_data, random_seed)
        # TODO remove the provenance columns here

        print("Encoding training data")
        feature_transformer = ctx.encode_features_function()
        model_training = ctx.model_training_function()

        model = Pipeline(stages=[feature_transformer, model_training]).fit(intermediate_train)

        predictions = model.transform(intermediate_test)

        predictions_and_labels = predictions.select(['prediction', 'label']).rdd \
            .map(lambda row: (row['prediction'], float(row['label'])))

        metrics = MulticlassMetrics(predictions_and_labels)
        print(f'Accuracy: {metrics.accuracy}')

    else:
        print("Splitting prepared data")
        intermediate_train, intermediate_test = ctx.split_function(intermediate, random_seed)

        prov_columns = ', '.join(prepared_data.duckframe.provenance_columns)
        train_provenance = duckdb.query(f"SELECT {prov_columns} FROM intermediate_train").to_df()
        train_df = duckdb.query(f"SELECT * EXCLUDE {prov_columns} FROM intermediate_train").to_df()

        test_provenance = duckdb.query(f"SELECT {prov_columns} FROM intermediate_test").to_df()
        test_df = duckdb.query(f"SELECT * EXCLUDE {prov_columns} FROM intermediate_test").to_df()

        print("Encoding training data")
        feature_transformer = ctx.encode_features_function()
        target_encoder = ctx.encode_target_function()
        target_column = ctx.encode_target_column

        X_train = feature_transformer.fit_transform(train_df)
        y_train = target_encoder.fit_transform(train_df[target_column])

        print("Executing model training")
        model = ctx.model_training_function()
        model.fit(X_train, y_train)

        print("Encoding test data")
        X_test = feature_transformer.transform(test_df)
        y_test = target_encoder.transform(test_df[target_column])

        print("Evaluating the model on test data")
        score = model.score(X_test, y_test)

        print("Score", score)
