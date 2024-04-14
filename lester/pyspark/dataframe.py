from pyspark.sql.functions import monotonically_increasing_id


class TrackedDataframe:

    def __init__(self, df, source_id=None):
        if source_id is None:
            self.df = df
        else:
            prov_column = f'__lester_provenance__{source_id}'
            self.df = df.withColumn(prov_column, monotonically_increasing_id())

    def filter(self, expression):
        return TrackedDataframe(self.df.filter(expression))

    def dropna(self):
        return TrackedDataframe(self.df.dropna())

    def join(self, other, on):
        # TODO we might get issues with self joins here
        return TrackedDataframe(self.df.join(other.df, on=on))

    def select(self, expression):
        return TrackedDataframe(self.df.select(expression))

    def withColumn(self, new_column, expression):
        return TrackedDataframe(self.df.withColumn(new_column, expression))

    def randomSplit(self, weights, seed):
        return [TrackedDataframe(split) for split in self.df.randomSplit(weights, seed)]

    def __getattr__(self, name):
        return self.df.__getattr__(name)
