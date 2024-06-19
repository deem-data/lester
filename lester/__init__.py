import pandas as pd
import zlib

GLOBAL_VARIABLES_FROM_CALLER = {}
LOCAL_VARIABLES_FROM_CALLER = {}


# TODO check if there is some trick to avoid this...
def make_accessible(local_variables, global_variables):
    global GLOBAL_VARIABLES_FROM_CALLER
    GLOBAL_VARIABLES_FROM_CALLER = global_variables
    global LOCAL_VARIABLES_FROM_CALLER
    LOCAL_VARIABLES_FROM_CALLER = local_variables


def read_csv(path, header, names):
    df = pd.read_csv(path, header=header, names=names)
    source_name = hex(zlib.crc32(str.encode(path)))
    column_provenance = {
        column: [f"{source_name}.{column}"] for column in df.columns
    }
    row_provenance_column = f'__lester_provenance_{source_name}'
    df[row_provenance_column] = range(len(df))
    row_provenance_columns = [row_provenance_column]

    return TrackedDataframe(source_name, df, row_provenance_columns, column_provenance)


def join(left_df, right_df, left_on, right_on):
    return left_df.join(right_df, left_on, right_on)


class TrackedDataframe:

    def __init__(self, source_name, df, row_provenance_columns, column_provenance):
        self.df = df
        self.source_name = source_name
        self.row_provenance_columns = row_provenance_columns
        self.column_provenance = column_provenance

    def __len__(self):
        return len(self.df)

    def join(self, other, left_on, right_on):
        # TODO handle name collisions here
        result_row_provenance_columns = self.row_provenance_columns + other.row_provenance_columns
        result_column_provenance = {**self.column_provenance, **other.column_provenance}
        result_df = self.df.merge(other.df, left_on=left_on, right_on=right_on)
        return TrackedDataframe(self.source_name, result_df, result_row_provenance_columns, result_column_provenance)

    def filter(self, predicate_expression):
        result_row_provenance_columns = self.row_provenance_columns
        result_column_provenance = self.column_provenance
        result_df = self.df.query(predicate_expression,
                                  local_dict=LOCAL_VARIABLES_FROM_CALLER, global_dict=GLOBAL_VARIABLES_FROM_CALLER)
        return TrackedDataframe(self.source_name, result_df, result_row_provenance_columns, result_column_provenance)

    def __getitem__(self, columns):
        result_row_provenance_columns = self.row_provenance_columns
        result_column_provenance = {column: provenance for column, provenance in self.column_provenance.items()
                                    if column in columns}
        target_columns = columns + self.row_provenance_columns
        result_df = self.df[target_columns]
        return TrackedDataframe(self.source_name, result_df, result_row_provenance_columns, result_column_provenance)

    def project(self, target_column, source_columns, func):
        result_row_provenance_columns = self.row_provenance_columns
        result_column_provenance = self.column_provenance.copy()
        result_column_provenance[target_column] = [f"{self.source_name}.{column}" for column in source_columns]

        target_column_values = []
        for _, row in self.df.iterrows():
            target_column_values.append(func(row))

        result_df = self.df.copy(deep=True)
        result_df[target_column] = target_column_values

        return TrackedDataframe(self.source_name, result_df, result_row_provenance_columns, result_column_provenance)
