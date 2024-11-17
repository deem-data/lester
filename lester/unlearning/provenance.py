class ProvenanceQueries:

    def __init__(self, artifacts):
        self.artifacts = artifacts

    def train_rows_originating_from(self, source_name, provenance_identifiers):
        train_prov = self.artifacts.load_train_provenance()
        prov_column = self.artifacts.provenance_column_name(source_name)
        row_indexes = train_prov.index[train_prov[prov_column].isin(provenance_identifiers)].tolist()
        return row_indexes

    def test_rows_originating_from(self, source_name, provenance_identifiers):
        test_prov = self.artifacts.load_test_provenance()
        prov_column = self.artifacts.provenance_column_name(source_name)
        row_indexes = test_prov.index[test_prov[prov_column].isin(provenance_identifiers)].tolist()
        return row_indexes

    def output_columns(self, source_name, source_column_name):
        source_column = f'{source_name}.{source_column_name}'
        output_columns = [column
                          for column, source_columns in self.artifacts.column_provenance.items()
                          if source_column in source_columns and column != 'is_highly_rated']  # This is due a bug...
        return output_columns

    def feature_ranges(self, output_columns):
        feature_ranges = [rnge for column, rnge
                          in self.artifacts.matrix_column_provenance.items()
                          if column in output_columns]
        return feature_ranges
