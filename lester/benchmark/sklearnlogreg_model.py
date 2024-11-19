import torch.nn

from lester.benchmark import ModelCodeTransformationTask


class SklearnLogisticRegressionTransformationTask(ModelCodeTransformationTask):
    @property
    def original_code(self):
        return """
from sklearn.linear_model import SGDClassifier
model = SGDClassifier(loss="log_loss", penalty=None)        
        """

    def run_manually_rewritten_code(self, params):
        import torch.nn as nn

        class LogisticRegressionModel(nn.Module):
            def __init__(self, input_dim):
                super(LogisticRegressionModel, self).__init__()
                self.linear = nn.Linear(input_dim, 1)

            def forward(self, x):
                return torch.sigmoid(self.linear(x))

        model = LogisticRegressionModel(params['num_features'])
        loss = torch.nn.BCELoss

        return model, loss

    def evaluate_transformed_code(self, transformed_code):
        pass