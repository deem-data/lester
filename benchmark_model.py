from lester.benchmark.sklearnlogreg_model import SklearnLogisticRegressionTransformationTask
from lester.benchmark.sklearnsvm_model import SklearnSVMTransformationTask

SKLEARNLOGREG_CODE = """
def __model(num_features):
    import torch
    import torch.nn as nn

    class LogisticRegressionModel(nn.Module):
        def __init__(self, num_features):
            super(LogisticRegressionModel, self).__init__()
            self.linear = nn.Linear(num_features, 1)

        def forward(self, x):
            return torch.sigmoid(self.linear(x))

    model = LogisticRegressionModel(num_features)
    loss = nn.BCELoss()

    return model, loss
"""

SKLEARNSVM_CODE = """
def __model(num_features):
    import torch
    import torch.nn as nn

    class LinearSVC(nn.Module):
        def __init__(self, num_features):
            super(LinearSVC, self).__init__()
            self.linear = nn.Linear(num_features, 1)

        def forward(self, x):
            return self.linear(x)

    model = LinearSVC(num_features)
    loss = nn.HingeEmbeddingLoss()

    return model, loss
"""

print("SklearnLogisticRegression...")
logregtask = SklearnLogisticRegressionTransformationTask()
logregtask.evaluate_transformed_code(SKLEARNLOGREG_CODE)

print("SklearnSVM...")
svmtask = SklearnSVMTransformationTask()
svmtask.evaluate_transformed_code(SKLEARNSVM_CODE)
