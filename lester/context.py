from lester.pandas import WrappedDataframe, TrackedDataframe

from dataclasses import dataclass
import duckdb

import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# TODO replace with a singleton
__LESTER_CONTEXT = {
    "prepare_function": None,
    "prepare_function_args": None,
    "split_function": None,
    "encode_features_function": None,
    "encode_target_function": None,
    "encode_target_column": None,
    "model_training_function": None,
    "source_counter": 0,
}


@dataclass
class datasource:
    name: str
    track_provenance: bool = False


def prepare(*args, **kwargs):
    def inner(func):
        __LESTER_CONTEXT["prepare_function"] = func
        __LESTER_CONTEXT["prepare_function_args"] = kwargs
        return func
    return inner


def split(*args, **kwargs):
    def inner(func):
        __LESTER_CONTEXT["split_function"] = func
        return func
    return inner


def encode_features(*args, **kwargs):
    def inner(func):
        __LESTER_CONTEXT["encode_features_function"] = func
        return func
    return inner


def encode_target(*args, **kwargs):
    def inner(func):
        __LESTER_CONTEXT["encode_target_function"] = func
        __LESTER_CONTEXT["encode_target_column"] = kwargs['target_column']
        return func
    return inner


def model_training(*args, **kwargs):
    def inner(func):
        __LESTER_CONTEXT["model_training_function"] = func
        return func
    return inner


def __load_lester_datasource(source):

    name_to_query = {
        'ratings': "SELECT * FROM 'data/ratings.csv'",
        'books': "SELECT * FROM 'data/books.csv'",
        'categories': "SELECT * FROM 'data/categories.csv'",
        'book_tags': "SELECT * FROM 'data/book_tags.csv'",
        'tags': "SELECT * FROM 'data/tags.csv'",
    }

    query = name_to_query[source.name]
    df = duckdb.query(query).to_df()
    if not source.track_provenance:
        return WrappedDataframe(df)
    else:
        source_id = __LESTER_CONTEXT['source_counter']
        __LESTER_CONTEXT['source_counter'] += 1
        return TrackedDataframe(df, source_id=source_id)


def run():
    random_seed = 42

    eprint("Loading data from data sources")
    datasouce_args = {}
    for kwarg_name, source in __LESTER_CONTEXT["prepare_function_args"].items():
        datasouce_args[kwarg_name] = __load_lester_datasource(source)

    eprint("Executing relational data preparation")
    prepared_data = __LESTER_CONTEXT["prepare_function"](**datasouce_args)

    prepared_data_df = prepared_data.df
    prepared_data_prov = prepared_data.provenance
    prov_columns = ', '.join(list(prepared_data_prov.columns))

    intermediate = duckdb.query(f"""
        SELECT * 
        FROM prepared_data_df 
        POSITIONAL JOIN prepared_data_prov
    """).to_df()

    eprint("Splitting prepared data")
    intermediate_train, intermediate_test = __LESTER_CONTEXT["split_function"](intermediate, random_seed)

    train_provenance = duckdb.query(f"SELECT {prov_columns} FROM intermediate_train").to_df()
    train_df = duckdb.query(f"SELECT * EXCLUDE {prov_columns} FROM intermediate_train").to_df()

    test_provenance = duckdb.query(f"SELECT {prov_columns} FROM intermediate_test").to_df()
    test_df = duckdb.query(f"SELECT * EXCLUDE {prov_columns} FROM intermediate_test").to_df()

    eprint("Encoding training data")
    feature_transformer = __LESTER_CONTEXT["encode_features_function"]()
    target_encoder = __LESTER_CONTEXT["encode_target_function"]()
    target_column = __LESTER_CONTEXT["encode_target_column"]

    X_train = feature_transformer.fit_transform(train_df)
    y_train = target_encoder.fit_transform(train_df[target_column])

    eprint("Shape of X_train", X_train.shape)

    eprint("Executing model training")
    model = __LESTER_CONTEXT["model_training_function"]()
    model.fit(X_train, y_train)

    eprint("Encoding test data")
    X_test = feature_transformer.fit_transform(train_df)
    y_test = target_encoder.fit_transform(train_df[target_column])

    eprint("Evaluating the model on test data")
    score = model.score(X_test, y_test)

    eprint("Score", score)