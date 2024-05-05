import duckdb


def from_tracked_source(name, path, primary_key_columns):
    column_expression = ', '.join(primary_key_columns)
    provenance_column = f"__lester_provenance_{name}"

    duckdb.execute(f"""
        CREATE OR REPLACE VIEW {name}_view AS
            SELECT 
                *, 
                ROW_NUMBER() OVER (ORDER BY {column_expression}) AS {provenance_column} 
            FROM '{path}'
        """)

    column_name_result = duckdb \
        .execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{name}_view'") \
        .fetchall()

    columns = [name[0] for name in column_name_result if name[0] != provenance_column]
    column_provenance = {column: [f"{name}.{column}"] for column in columns}

    relation = duckdb.sql(f"SELECT * FROM {name}_view") \
        .set_alias(name)

    return Duckframe(name, relation, columns, column_provenance, [provenance_column])


def from_source(name, path):

    duckdb.execute(f"""
        CREATE OR REPLACE VIEW {name}_view AS
            SELECT * FROM '{path}'
        """)

    column_name_result = duckdb.execute(f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = '{name}_view'
        """) \
        .fetchall()

    columns = [name[0] for name in column_name_result]
    column_provenance = {column: [f"{name}.{column}"] for column in columns}

    relation = duckdb.sql(f"SELECT * FROM {name}_view") \
        .set_alias(name)

    return Duckframe(name, relation, columns, column_provenance)


def empty_if_none(lst):
    if lst is None:
        return []
    else:
        return lst


class Duckframe:

    def __init__(self, name, relation, columns, column_provenance, provenance_columns=None):
        self.name = name
        self.relation = relation
        self.columns = columns
        self.column_provenance = column_provenance
        self.provenance_columns = provenance_columns
        self.consumer_count = 0

    def select(self, predicate):
        result_name = f'{self.name}__select_{self.consumer_count}'
        self.consumer_count += 1

        result_relation = self.relation \
            .filter(predicate) \
            .set_alias(result_name)

        return Duckframe(result_name, result_relation, self.columns, self.column_provenance, self.provenance_columns)

    def dropna(self):
        predicate = ' AND '.join([f'({column} IS NOT NULL)' for column in self.columns])
        return self.select(predicate)

    # TODO: this probably fails if we try overwrite an existing column
    def extended_project(self, new_column, column_expression):
        result_name = f'{self.name}__extended_project_{self.consumer_count}'
        self.consumer_count += 1

        projection_expression = f'*, {column_expression} AS {new_column}'

        result_relation = self.relation \
            .project(projection_expression) \
            .set_alias(result_name)

        result_columns = self.columns + [new_column]

        # TODO: we need a way to get the source columns from the expression
        result_column_provenance = {
            **self.column_provenance,
            new_column: [provenance_entry
                         for provenance in self.column_provenance.values()
                         for provenance_entry in provenance]
        }

        return Duckframe(result_name, result_relation, result_columns, result_column_provenance,
                         self.provenance_columns)

    def project(self, columns):
        result_name = f'{self.name}__project_{self.consumer_count}'
        self.consumer_count += 1

        projection_expression = ', '.join(columns + empty_if_none(self.provenance_columns))

        result_relation = self.relation \
            .project(projection_expression) \
            .set_alias(result_name)

        result_column_provenance = {column: provenance for (column, provenance) in self.column_provenance
                                    if column in columns}

        return Duckframe(result_name, result_relation, columns, result_column_provenance, self.provenance_columns)

    def join(self, other, on):
        result_name = f'{self.name}__join_{other.name}_{self.consumer_count}'
        self.consumer_count += 1

        join_condition = f'{self.name}.{on}={other.name}.{on}'

        result_relation = self.relation \
            .join(other.relation, join_condition) \
            .set_alias(result_name)

        # TODO needs some special handling for self-joins and duplicate column names...
        result_columns = self.columns + other.columns
        result_provenance_columns = empty_if_none(self.provenance_columns) + empty_if_none(other.provenance_columns)
        # TODO We need to read up on DuckDB's semantics when encountering duplicate column names
        result_column_provenance = {**self.column_provenance, **other.column_provenance}

        return Duckframe(result_name, result_relation, result_columns, result_column_provenance,
                         result_provenance_columns)
