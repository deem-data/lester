from lester.context import LesterContext
from lester.pandas.dataframe import WrappedDataframe, TrackedDataframe
import duckdb
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def __load_lester_datasource(source):

    ctx = LesterContext()

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
        source_id = ctx.source_counter
        ctx.source_counter += 1
        return TrackedDataframe(df, source_id=source_id)


def run():

    ctx = LesterContext()
    random_seed = 42

    eprint("Loading data from data sources")
    datasouce_args = {}
    for kwarg_name, source in ctx.prepare_function_args.items():
        datasouce_args[kwarg_name] = __load_lester_datasource(source)

    eprint("Executing relational data preparation")
    prepared_data = ctx.prepare_function(**datasouce_args)

    prepared_data_df = prepared_data.df
    prepared_data_prov = prepared_data.provenance
    prov_columns = ', '.join(list(prepared_data_prov.columns))

    intermediate = duckdb.query(f"""
        SELECT * 
        FROM prepared_data_df 
        POSITIONAL JOIN prepared_data_prov
    """).to_df()

    eprint("Splitting prepared data")
    intermediate_train, intermediate_test = ctx.split_function(intermediate, random_seed)

    train_provenance = duckdb.query(f"SELECT {prov_columns} FROM intermediate_train").to_df()
    train_df = duckdb.query(f"SELECT * EXCLUDE {prov_columns} FROM intermediate_train").to_df()

    test_provenance = duckdb.query(f"SELECT {prov_columns} FROM intermediate_test").to_df()
    test_df = duckdb.query(f"SELECT * EXCLUDE {prov_columns} FROM intermediate_test").to_df()

    eprint("Encoding training data")
    feature_transformer = ctx.encode_features_function()
    target_encoder = ctx.encode_target_function()
    target_column = ctx.encode_target_column

    X_train = feature_transformer.fit_transform(train_df)
    y_train = target_encoder.fit_transform(train_df[target_column])

    eprint("Shape of X_train", X_train.shape)

    eprint("Executing model training")
    model = ctx.model_training_function()
    model.fit(X_train, y_train)

    eprint("Encoding test data")
    X_test = feature_transformer.fit_transform(train_df)
    y_test = target_encoder.fit_transform(train_df[target_column])

    eprint("Evaluating the model on test data")
    score = model.score(X_test, y_test)

    eprint("Score", score)
