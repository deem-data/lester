from lester.context import model_training, EstimatorTransformerDialect

import numpy as np
import torch
from torch import nn
from skorch import NeuralNetBinaryClassifier
from skorch.dataset import Dataset


class MLP(nn.Module):
    def __init__(self, num_features):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


class MyDataset(Dataset):
    def transform(self, X, y):
        X_transformed = X.astype(np.float32)

        if y is not None:
            y_transformed = y.astype(np.float32)
        else:
            y_transformed = None

        return super().transform(X_transformed, y_transformed)


@model_training(dialect=EstimatorTransformerDialect.SKLEARN)
def custom_mlp():
    return NeuralNetBinaryClassifier(
        MLP(num_features=3672),
        max_epochs=10,
        lr=0.001,
        iterator_train__shuffle=True,
        criterion=torch.nn.BCELoss,
        optimizer=torch.optim.Adam,
        dataset=MyDataset
    )
