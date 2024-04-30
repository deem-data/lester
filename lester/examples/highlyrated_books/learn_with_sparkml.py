from lester.context import train_model, EstimatorTransformerDialect
from pyspark.ml.classification import LogisticRegression


@train_model(dialect=EstimatorTransformerDialect.SPARKML)
def logreg_with_fixed_hyperparams():
    return LogisticRegression(maxIter=10, regParam=0.001)
