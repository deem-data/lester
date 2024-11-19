DATAPREP_COT = """
The following code prepares data for a machine learning task and is written in Python with for loops and manual data parsing. 
Your task is to rewrite the code to a dataframe library called lester without changing the code semantics.

The lester dataframe library has an API similar to pandas and supports the following dataframe operations:

 * ``read_csv(path, header, names, sep, parse_dates)`` which has the same semantics and arguments as the ``read_csv`` operation from pandas.
 * ``join(left_df, right_df, left_on, right_on)`` which has the same semantics and arguments as the ``merge`` operation from pandas.
 * ``union(list_of_dataframes)`` which has the same semantics as the ``concat`` operation from pandas.
 * ``filter(self, predicate_expression)`` which has the same semantics as the ``query`` operation from pandas. This operation is directly invoked on the dataframe object.
 * ``rename(self, column_mapping)`` which has the same semantics as the ``rename`` operation from pandas. This operation is directly invoked on the dataframe object.
 * ``__getitem__(self, columns)`` which has the same semantics as the ``__getitem__`` operation from pandas. This operation is directly invoked on the dataframe object.
 * ``project(self, target_column, source_columns, func)`` which is similar to the assign operation from pandas, but has two additional parameters: `target_column` and `source_columns`; `target_column` refers to the new column which should be created, while `source_columns` refers to the list of input columns that would be used by the `func` expression in `assign`. This operation is directly invoked on the dataframe object. 
  
The lester library DOES NOT OFFER ANY OTHER METHODS!

Proceed in the following way:
1. Import the lester library via ``import lester as ld``. Use it for the rest of the code, similar to how we would use pandas.
2. Rewrite the python code to use the lester dataframe API without changing code semantics. {input_hint}
3. Rewrite the python code such that each new column is created with a separate ``project`` statement. Each function that is used as ``func`` in the ``project`` statement must be able to work on a single value or (list of values) of a row in the dataframe.
4. Rewrite the python code such that all the code is contained in a single function with the name and signature ``__dataprep({input_args})``.
5. Rewrite the python code such that global variables and imports are moved into the `__dataprep`` function. Make sure that no imports from the original code are missing!
6. Rewrite the python code to return a single dataframe called `result_df` as result. This final dataframe should have the following columns: {columns}

ONLY RESPOND WITH PYTHON CODE. DO NOT CHANGE THE SEMANTICS OF THE CODE.

Here is the code to rewrite:

{code}
"""

FIX = """
The generated code did not execute correctly, but produced the following error:

{error_message}

Please rewrite the code to fix this error. Here is the code with the error that needs to be rewritten:

{generated_code}

ONLY RESPOND WITH PYTHON CODE. DO NOT CHANGE THE SEMANTICS OF THE CODE. RETURN THE FULL CODE FOR THE TASK!!!
"""

FEATURISATION_COT = """
The following code featurises a dataframe for a machine learning task and is written in Python. 
Your task is to rewrite the code to use the scikit-learn API without changing the code semantics.

Proceed in the following way:
1. Rewrite the python code such that all feature encoding operations use the Estimator/Transformer API from scikit-learn. 
2. Replace handwritten featurisation code with the corresponding Estimator/Transformer implementations from scikit-learn or generate new Estimator/Transformers if necessary.
3. Rewrite the python code such that all the code is contained in a single function with the name and signature ``__featurise()``. The goal of this function is to featurise a dataframe with the following columns: {columns}. This function has no arguments!
5. Rewrite the python code such that global variables and imports are moved into the ``__featurise`` function. Make sure that no imports from the original code are missing!
6. Rewrite the python code to use the ColumnTransformer from scikit-learn to combine all features. Make sure that the ``columns`` argument (the third value in the tuples supplied as the ``transformers`` parameter) is either a string or a list of strings, depending on the chosen Estimator/Transformer.
7. Return an unfitted instance of this ColumnTransformer from the ``__featurise`` function.

IMPORTANT: IGNORE CODE which prepares the label. This code will be handled separately. DO NOT MODIFY THE 

ONLY RESPOND WITH PYTHON CODE. DO NOT CHANGE THE SEMANTICS OF THE CODE.

Here is the code to rewrite:

{code}
"""

MODEL_COT = """
The following code defines a machine learning model from a common library in Python. 
Your task is to rewrite the code to use the pytorch API without changing the code semantics.

Proceed in the following way:
1. Rewrite the python code such that all the code is contained in a single function with the name and signature ``__model(num_features)``. This function returns a tuple (model, loss) containing a pytorch model and an appropriate loss function from pytorch. The parameter ``num_features`` denotes the number of input features for the model.
2. Rewrite the python code such that global variables and imports are moved into the `__model`` function. Make sure that no imports from the original code are missing!

ONLY RESPOND WITH PYTHON CODE. DO NOT CHANGE THE SEMANTICS OF THE CODE.

Here is the code to rewrite:

{code}
"""