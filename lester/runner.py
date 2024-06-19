import duckdb
import uuid
import os
import numpy as np
import json
import torch

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from lester.context import LesterContext, DataframeDialect, EstimatorTransformerDialect
from lester.tracked_dataframe.duckframe import from_tracked_source, from_source
from lester.tracked_dataframe.pandas import PandasDuckframe
from lester.tracked_dataframe.pyspark import PysparkDuckframe
from lester.tracked_dataframe.polars import PolarsDuckframe


def _find_dimensions(transformation, num_columns_transformed, start_index, end_index):
    if isinstance(transformation, StandardScaler):
        return [slice(start_index + index, start_index + index + 1)
                for index in range(0, num_columns_transformed)]
    elif isinstance(transformation, OneHotEncoder):
        ranges = []
        # TODO We should also include the drop and infrequent features in the calculation
        indices = list([0]) + list(np.cumsum([len(categories) for categories in transformation.categories_]))
        for offset in range(0, len(indices)-1):
            ranges.append(slice(indices[offset], indices[offset+1]))
        return ranges
    elif isinstance(transformation, Pipeline):
        _, last_transformation = transformation.steps[-1]
        # TODO We should also look at the intermediate steps of the pipeline in more elaborate cases
        return _find_dimensions(last_transformation, num_columns_transformed, start_index, end_index)
    elif isinstance(transformation, FunctionTransformer):
        # TODO check if the function transformer uses more than one column as input
        return [slice(start_index, end_index)]
    else:
        raise Exception(f'Cannot handle transformation: {transformation}')


def _matrix_column_provenance(column_transformer):
    matrix_column_provenance = []

    for transformer in column_transformer.transformers_:
        name, transformation, columns = transformer
        if name != 'remainder':
            start_index = column_transformer.output_indices_[name].start
            end_index = column_transformer.output_indices_[name].stop
            num_columns_transformed = len(columns)
            ranges = _find_dimensions(transformation, num_columns_transformed, start_index, end_index)

            if ranges is not None:
                matrix_column_provenance += list(zip(columns, ranges))

    return dict(matrix_column_provenance)


# TODO this should be refactored into a spark-specific class
def _matrix_column_provenance_spark(feature_transformer):
    from pyspark.ml.feature import OneHotEncoderModel, StringIndexerModel, VectorAssembler, HashingTF
    import networkx as nx

    G = nx.DiGraph()

    for transformer in reversed(feature_transformer.stages):

        inputs = []
        outputs = []
        output_sizes = None
        is_merge = False

        if isinstance(transformer, VectorAssembler):
            is_merge = True

        if transformer.hasParam('inputCol') and transformer.isSet('inputCol'):
            inputs = [transformer.getOrDefault('inputCol')]
        if transformer.hasParam('inputCols') and transformer.isSet('inputCols'):
            inputs = transformer.getOrDefault('inputCols')

        if transformer.hasParam('outputCol') and transformer.isSet('outputCol'):
            outputs = [transformer.getOrDefault('outputCol')]
        if transformer.hasParam('outputCols') and transformer.isSet('outputCols'):
            outputs = transformer.getOrDefault('outputCols')

        if isinstance(transformer, HashingTF):
            output_sizes = [transformer.getOrDefault('numFeatures')]
        if isinstance(transformer, OneHotEncoderModel):
            output_sizes = [size-1 for size in transformer.categorySizes]
        if isinstance(transformer, StringIndexerModel):
            output_sizes = [len(labels) for labels in transformer.labelsArray]

        for input_column in inputs:
            G.add_node(input_column)
        for output_column in outputs:
            G.add_node(output_column)

        if not is_merge:
            if output_sizes is None:
                for input_column, output_column in zip(inputs, outputs):
                    G.add_edge(input_column, output_column)
            else:
                for input_column, output_column, size in zip(inputs, outputs, output_sizes):
                    G.add_edge(input_column, output_column)
                    G[input_column][output_column]['size'] = size

        else:
            output_column = outputs[0]
            for input_column in inputs:
                G.add_edge(input_column, output_column)

    def walk_path(node, G, path, last_size):
        if G.out_degree(node) == 0:
            return path, last_size
        else:
            successor = list(G.successors(node))[0]

            if 'size' in G[node][successor]:
                last_size = G[node][successor]['size']

            order = 0
            for index, sucessor_predecessor in enumerate(G.predecessors(successor)):
                if sucessor_predecessor == node:
                    order = index

            path.append((successor, order))

            return walk_path(successor, G, path, last_size)

    sources = [node for node in G.nodes() if G.in_degree(node) == 0]

    mappings = []
    for source in sources:
        path, size = walk_path(source, G, [], 1)
        orders = list(reversed([order for _, order in path]))
        mappings.append((source, size, orders))

    sorted_mappings = sorted(mappings, key=lambda x: x[2])

    matrix_column_provenance = {}
    index = 0
    for source, size, _ in sorted_mappings:
        matrix_column_provenance[source] = slice(index, index + size)
        index = index + size

    return matrix_column_provenance


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


def matrix_column_provenance_as_json(matrix_column_provenance):
    serialized_dict = {}
    for key, value in matrix_column_provenance.items():
        serialized_dict[key] = (int(value.start), int(value.stop))
    return serialized_dict


def _save_as_json(file, python_dict):
    with open(file, 'w') as f:
        json.dump(python_dict, f, indent=2)


def _persist_with_row_provenance(intermediate_train, intermediate_test, prov_columns, artifact_path):

    print("Persisting relational data")
    train = duckdb.query(f"SELECT * EXCLUDE({prov_columns}) FROM intermediate_train").to_df()
    test = duckdb.query(f"SELECT * EXCLUDE ({prov_columns}) FROM intermediate_test").to_df()
    train.to_parquet(f'{artifact_path}/train.parquet', index=False)
    test.to_parquet(f'{artifact_path}/test.parquet', index=False)

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
        # TODO this must be configurable
        spark = SparkSession.builder \
            .master("local[4]") \
            .config("spark.driver.bindAddress", "127.0.0.1") \
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

        _persist_with_row_provenance(intermediate_train.toPandas(), intermediate_test.toPandas(),
                                     prov_columns, artifact_path)
        # TODO remove the provenance columns here?

        print("Encoding training data")
        feature_transformer = ctx.encode_features_function()
        model_training = ctx.model_training_function()

        fitted_feature_transformer = feature_transformer.fit(intermediate_train)

        matrix_column_provenance = _matrix_column_provenance_spark(fitted_feature_transformer)
        _save_as_json(f'{artifact_path}/matrix_column_provenance.json',
                      matrix_column_provenance_as_json(matrix_column_provenance))
        print(matrix_column_provenance)

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

        _persist_with_row_provenance(intermediate_train, intermediate_test, prov_columns, artifact_path)
        train_df = duckdb.query(f"SELECT * EXCLUDE {prov_columns} FROM intermediate_train").to_df()
        test_df = duckdb.query(f"SELECT * EXCLUDE {prov_columns} FROM intermediate_test").to_df()



        print("Encoding training data")
        feature_transformer = ctx.encode_features_function()
        target_encoder = ctx.encode_target_function()
        target_column = ctx.encode_target_column

        X_train = feature_transformer.fit_transform(train_df)
        y_train = target_encoder.fit_transform(train_df[target_column])

        matrix_column_provenance = _matrix_column_provenance(feature_transformer)
        _save_as_json(f'{artifact_path}/matrix_column_provenance.json',
                      matrix_column_provenance_as_json(matrix_column_provenance))

        print("Executing model training")
        num_features = X_train.shape[1]
        model = ctx.model_training_function(num_features)
        model.fit(X_train, y_train)

        print("Encoding test data")
        X_test = feature_transformer.transform(test_df)
        y_test = target_encoder.transform(test_df[target_column])

        print("Evaluating the model on test data")
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        print("Score", score)

        _persist_matrices(X_train, y_train, X_test, y_test, y_pred, artifact_path)
        torch.save(model.module_, f"{artifact_path}/model.pt")

    return run_id

