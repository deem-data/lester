import duckdb
import uuid
import os
import numpy as np
import json
from sklearn.metrics import accuracy_score

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


def _save_as_json(file, python_dict):
    with open(file, 'w') as f:
        json.dump(python_dict, f, indent=2)


def _persist_row_provenance(intermediate_train, intermediate_test, prov_columns, artifact_path):
    print("Persisting provenance")
    row_provenance_X_train = duckdb.query(f"SELECT {prov_columns} FROM intermediate_train").to_df()
    row_provenance_X_test = duckdb.query(f"SELECT {prov_columns} FROM intermediate_test").to_df()

    row_provenance_X_train.to_parquet(f'{artifact_path}/row_provenance_X_train.parquet', index=False)
    row_provenance_X_test.to_parquet(f'{artifact_path}/row_provenance_X_test.parquet', index=False)


def _persist_matrices(X_train, y_train, X_test, y_test, y_pred, artifact_path):
    np.save(f'{artifact_path}/X_train.npy', X_train)
    np.save(f'{artifact_path}/y_train.npy', y_train)
    np.save(f'{artifact_path}/X_test.npy', X_test)
    np.save(f'{artifact_path}/y_test.npy', y_test)
    np.save(f'{artifact_path}/y_pred.npy', y_pred)


def run_pipeline(name, source_paths, random_seed=42):

    run_id = uuid.uuid4()
    artifact_path = f'.lester/{name}/{run_id}'
    os.makedirs(artifact_path)

    print(f"Starting lester run ({artifact_path})")
    _save_as_json(f'{artifact_path}/source_paths.json', source_paths)

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

    column_provenance = prepared_data.duckframe.column_provenance
    _save_as_json(f'{artifact_path}/column_provenance.json', column_provenance)

    prov_columns = ', '.join(prepared_data.duckframe.provenance_columns)

    # TODO We don't handle more crazy combinations at the moment
    if ctx.model_training_dialect == EstimatorTransformerDialect.SPARKML and \
            ctx.encode_features_dialect == EstimatorTransformerDialect.SPARKML:

        from pyspark.ml import Pipeline
        from pyspark.mllib.evaluation import MulticlassMetrics

        prepared_data = spark.createDataFrame(intermediate)

        print("Splitting prepared data")
        intermediate_train, intermediate_test = ctx.split_function(prepared_data, random_seed)

        _persist_row_provenance(intermediate_train.toPandas(), intermediate_test.toPandas(),
                                prov_columns, artifact_path)
        # TODO remove the provenance columns here?

        print("Encoding training data")
        feature_transformer = ctx.encode_features_function()
        model_training = ctx.model_training_function()

        fitted_feature_transformer = feature_transformer.fit(intermediate_train)

        X_y_train = fitted_feature_transformer.transform(intermediate_train)
        X_y_train_df = X_y_train.select(['features', 'label']).toPandas()
        X_train = np.vstack(X_y_train_df['features'].apply(lambda x: x.toArray()).values)
        y_train = np.array(X_y_train_df['label'].values)

        model = model_training.fit(X_y_train)

        X_y_test = fitted_feature_transformer.transform(intermediate_test)
        X_y_test_df = X_y_test.select(['features', 'label']).toPandas()
        X_test = np.vstack(X_y_test_df['features'].apply(lambda x: x.toArray()).values)
        y_test = np.array(X_y_test_df['label'].values)

        predictions = model.transform(X_y_test)

        predictions_df = predictions.select('prediction').toPandas()
        y_pred = np.array(predictions_df['prediction'].values)

        predictions_and_labels = predictions.select(['prediction', 'label']).rdd \
            .map(lambda row: (row['prediction'], float(row['label'])))

        metrics = MulticlassMetrics(predictions_and_labels)
        print(f'Accuracy: {metrics.accuracy}')

        _persist_matrices(X_train, y_train, X_test, y_test, y_pred, artifact_path)

    else:
        print("Splitting prepared data")
        intermediate_train, intermediate_test = ctx.split_function(intermediate, random_seed)

        _persist_row_provenance(intermediate_train, intermediate_test, prov_columns, artifact_path)
        train_df = duckdb.query(f"SELECT * EXCLUDE {prov_columns} FROM intermediate_train").to_df()
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
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        print("Score", score)

        _persist_matrices(X_train, y_train, X_test, y_test, y_pred, artifact_path)
