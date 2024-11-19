from lester.benchmark import ModelCodeTransformationTask


class SklearnSVMTransformationTask(ModelCodeTransformationTask):
    @property
    def original_code(self):
        return """
from sklearn import svm
model = svm.LinearSVC(penalty='l2', loss='squared_hinge')
        """

    def run_manually_rewritten_code(self, params):
        import torch
        import torch.nn as nn

        class LinearSVC(torch.nn.Module):
            def __init__(self, num_features):
                super(LinearSVC, self).__init__()
                self.linear = nn.Linear(num_features, 1)

            def forward(self, x):
                return self.linear(x)

        model = LinearSVC(params['num_features'])
        loss = nn.HingeEmbeddingLoss()  # Squared hinge loss is not directly available, so hinge loss is used.
        return model, loss

    def evaluate_transformed_code(self, transformed_code):
        pass