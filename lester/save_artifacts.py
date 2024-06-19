import duckdb
import json
import numpy as np


def _persist_matrices(X_train, y_train, X_test, y_test, y_pred, artifact_path):
    np.save(f'{artifact_path}/X_train.npy', X_train)
    np.save(f'{artifact_path}/y_train.npy', y_train)
    np.save(f'{artifact_path}/X_test.npy', X_test)
    np.save(f'{artifact_path}/y_test.npy', y_test)
    np.save(f'{artifact_path}/y_pred.npy', y_pred)


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


def matrix_column_provenance_as_json(matrix_column_provenance):
    serialized_dict = {}
    for key, ranges in matrix_column_provenance.items():
        serialized_dict[key] = [(int(value.start), int(value.stop)) for value in ranges]
    return serialized_dict


def _save_as_json(file, python_dict):
    with open(file, 'w') as f:
        json.dump(python_dict, f, indent=2)
