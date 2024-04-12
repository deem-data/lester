import pandas as pd
import duckdb


class WrappedDataframe:

    def __init__(self, df):
        self.df = df

    def query(self, predicate):
        self_df = self.df

        result_df = duckdb.query(f"""
            SELECT * 
            FROM self_df 
            WHERE {predicate}
        """).to_df()

        return WrappedDataframe(result_df)

    def merge(self, other, on):
        if isinstance(other, TrackedDataframe):
            raise("NA")
        else:
            self_df = self.df
            other_df = other.df

            query = f"""
            SELECT s.*, o.*
            FROM self_df s 
            JOIN other_df o ON s.{on} = o.{on}
            """

        intermediate = duckdb.query(query).to_df()
        return WrappedDataframe(intermediate)

    def dropna(self):
        predicate = ' AND '.join([f"{column} IS NOT NULL" for column in self.df.columns])
        return self.query(predicate)

    def __getitem__(self, columns):
        self_df = self.df

        if isinstance(columns, list):
            column_expression = ', '.join(columns)
        else:
            column_expression = columns

        result_df = duckdb.query(f"SELECT {column_expression} FROM self_df").to_df()
        return WrappedDataframe(result_df)

    def __getattr__(self, attribute_name):
        if attribute_name == 'columns':
            return self.df.columns


class TrackedDataframe:

    def __init__(self, df, source_id=None, provenance=None):
        self.df = df

        if provenance is not None:
            self.provenance = provenance
        else:
            if source_id is None:
                raise Exception("Need source_id or provenance here!")
            else:
                self.provenance = pd.DataFrame({
                    f'__lester_provenance__{source_id}': [row_id for row_id in range(len(df))]
                })

    def query(self, predicate):
        self_df = self.df
        self_prov = self.provenance
        prov_columns = ', '.join(list(self_prov.columns))

        intermediate = duckdb.query(f"""
            SELECT * 
            FROM self_df 
            POSITIONAL JOIN self_prov
            WHERE {predicate}
        """).to_df()

        result_provenance = duckdb.query(f"SELECT {prov_columns} FROM intermediate").to_df()
        result_df = duckdb.query(f"SELECT * EXCLUDE ({prov_columns}) FROM intermediate").to_df()

        return TrackedDataframe(result_df, provenance=result_provenance)

    def dropna(self):
        predicate = ' AND '.join([f"{column} IS NOT NULL" for column in self.df.columns])
        return self.query(predicate)

    def assign(self, **kwargs):
        updated_df = self.df.assign(**kwargs)
        return TrackedDataframe(updated_df, provenance=self.provenance)

    def merge(self, other, on):
        if isinstance(other, TrackedDataframe):
            return self.__merge_tracked(other, on)
        else:
            return self.__merge_nontracked(other, on)

    def __merge_nontracked(self, other, on):
        self_df = self.df
        self_prov = self.provenance

        other_df = other.df

        prov_columns = ', '.join(list(self.provenance.columns))

        query = f"""
            WITH self_with_provenance AS (
              SELECT * 
                FROM self_df 
                POSITIONAL JOIN self_prov
            )
            
            SELECT s.*, o.*
            FROM self_with_provenance s 
            JOIN other_df o ON s.{on} = o.{on}
            """

        intermediate = duckdb.query(query).to_df()

        result_provenance = duckdb.query(f"SELECT {prov_columns} FROM intermediate").to_df()
        result_df = duckdb.query(f"SELECT * EXCLUDE {prov_columns} FROM intermediate").to_df()

        return TrackedDataframe(result_df, provenance=result_provenance)

    def __merge_tracked(self, other, on):
        self_df = self.df
        self_prov = self.provenance

        other_df = other.df
        other_prov = other.provenance

        # TODO this may not handle self joins correctly yet, will need some renaming
        prov_columns = ', '.join(list(self.provenance.columns) + list(other.provenance.columns))

        query = f"""
        WITH self_with_provenance AS (
          SELECT * 
            FROM self_df 
            POSITIONAL JOIN self_prov
        ),
        
        other_with_provenance AS (
          SELECT * 
            FROM other_df 
            POSITIONAL JOIN other_prov
        )
        
        SELECT s.*, o.*
        FROM self_with_provenance s 
        JOIN other_with_provenance o ON s.{on} = o.{on}
        """

        intermediate = duckdb.query(query).to_df()

        result_provenance = duckdb.query(f"SELECT {prov_columns} FROM intermediate").to_df()
        result_df = duckdb.query(f"SELECT * EXCLUDE {prov_columns} FROM intermediate").to_df()

        return TrackedDataframe(result_df, provenance=result_provenance)

    def __getitem__(self, columns):
        self_df = self.df

        if isinstance(columns, list):
            column_expression = ', '.join(columns)
        else:
            column_expression = columns

        result_df = duckdb.query(f"SELECT {column_expression} FROM self_df").to_df()
        return TrackedDataframe(result_df, provenance=self.provenance)

    def __getattr__(self, attribute_name):
        # TODO filter for provenance columns?
        if attribute_name == 'columns':
            return self.df.columns
