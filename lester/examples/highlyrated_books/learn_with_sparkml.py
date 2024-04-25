from lester.context import model_training, EstimatorTransformerDialect
from pyspark.ml.classification import LogisticRegression


@model_training(dialect=EstimatorTransformerDialect.SPARKML)
def logreg_with_fixed_hyperparams():
    return LogisticRegression(maxIter=10, regParam=0.001)
