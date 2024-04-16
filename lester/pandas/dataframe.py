class PandasDuckframe:

    def __init__(self, duckframe):
        self.duckframe = duckframe

    def query(self, predicate):
        result_duckframe = self.duckframe.select(predicate)
        return PandasDuckframe(result_duckframe)

    def dropna(self):
        result_duckframe = self.duckframe.dropna()
        return PandasDuckframe(result_duckframe)

    def merge(self, other, on):
        result_duckframe = self.duckframe.join(other.duckframe, on)
        return PandasDuckframe(result_duckframe)

    def eval(self, expression):
        return expression

    # TODO: this is a bit ugly, since it updates the instance itself...
    def __setitem__(self, column, eval_expression):
        result_duckframe = self.duckframe.extended_project(column, eval_expression)
        self.duckframe = result_duckframe

    def __getitem__(self, columns):
        result_duckframe = self.duckframe.project(columns)
        return PandasDuckframe(result_duckframe)
