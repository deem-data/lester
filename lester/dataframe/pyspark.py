class PysparkDuckframe:

    def __init__(self, duckframe):
        self.duckframe = duckframe

    def filter(self, predicate):
        result_duckframe = self.duckframe.select(predicate)
        return PysparkDuckframe(result_duckframe)

    def dropna(self):
        result_duckframe = self.duckframe.dropna()
        return PysparkDuckframe(result_duckframe)

    def join(self, other, on):
        result_duckframe = self.duckframe.join(other.duckframe, on)
        return PysparkDuckframe(result_duckframe)

    def select(self, columns):
        result_duckframe = self.duckframe.project(columns)
        return PysparkDuckframe(result_duckframe)

    def withColumn(self, new_column, column_expression):
        sql_expression = column_expression._jc.toString()
        result_duckframe = self.duckframe.extended_project(new_column, sql_expression)
        return PysparkDuckframe(result_duckframe)


