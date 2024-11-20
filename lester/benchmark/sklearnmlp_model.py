from lester.benchmark import ModelCodeTransformationTask


class SklearnMLPTransformationTask(ModelCodeTransformationTask):
    @property
    def original_code(self):
        return """
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=[64, 16], activation='relu')
        """

    def run_manually_rewritten_code(self, params):
        import torch
        import torch.nn as nn

        class CustomModel(nn.Module):
            def __init__(self, input_size):
                super(CustomModel, self).__init__()
                self.hidden1 = nn.Linear(input_size, 64)
                self.hidden2 = nn.Linear(64, 16)
                self.output = nn.Linear(16, 1)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.hidden1(x))
                x = self.relu(self.hidden2(x))
                x = self.output(x)
                return x

        model = CustomModel(params['num_features'])
        loss = nn.BCEWithLogitsLoss()

        return model, loss

    def evaluate_transformed_code(self, transformed_code):
        pass