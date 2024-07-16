import numpy as np
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


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

    merged_column_provenance = {}
    for column, rnge in matrix_column_provenance:
        if column not in merged_column_provenance:
            merged_column_provenance[column] = []
        merged_column_provenance[column].append(rnge)

    return merged_column_provenance


def _find_dimensions(transformation, num_columns_transformed, start_index, end_index):
    if isinstance(transformation, StandardScaler):
        return [slice(start_index + index, start_index + index + 1)
                for index in range(0, num_columns_transformed)]
    elif isinstance(transformation, OneHotEncoder):
        ranges = []
        # TODO We should also include the drop and infrequent features in the calculation
        indices = list([0]) + list(np.cumsum([len(categories) for categories in transformation.categories_]))
        for offset in range(0, len(indices) - 1):
            ranges.append(slice(indices[offset], indices[offset + 1]))
        return ranges
    elif isinstance(transformation, Pipeline):
        _, last_transformation = transformation.steps[-1]
        # TODO We should also look at the intermediate steps of the pipeline in more elaborate cases
        return _find_dimensions(last_transformation, num_columns_transformed, start_index, end_index)
    elif isinstance(transformation, FunctionTransformer):
        # TODO check if the function transformer uses more than one column as input
        return [slice(start_index, end_index)]
    else:
        # Probably a custom transformer, we need to handle it like the function transformer
        return [slice(start_index, end_index)]