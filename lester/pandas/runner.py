from lester.context import LesterContext
from lester.duckframe import from_tracked_source, from_source
from lester.pandas.dataframe import PandasDuckframe
import duckdb
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

    return PandasDuckframe(duckframe)


def run():

    ctx = LesterContext()
    random_seed = 42

    eprint("Loading data from data sources")
    datasouce_args = {}
    for kwarg_name, source in ctx.prepare_function_args.items():
        datasouce_args[kwarg_name] = __load_lester_datasource(source)

    eprint("Executing relational data preparation")
    prepared_data = ctx.prepare_function(**datasouce_args)
    intermediate = prepared_data.duckframe.relation.to_df()
    prov_columns = ', '.join(prepared_data.duckframe.provenance_columns)

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
    X_test = feature_transformer.transform(test_df)
    y_test = target_encoder.transform(test_df[target_column])

    eprint("Evaluating the model on test data")
    score = model.score(X_test, y_test)

    eprint("Score", score)
