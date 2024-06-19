import torch
from torch import nn
from skorch import NeuralNetBinaryClassifier


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


def custom_mlp(num_features):
    return NeuralNetBinaryClassifier(
        MLP(num_features=num_features),
        max_epochs=25,
        lr=0.001,
        iterator_train__shuffle=True,
        criterion=torch.nn.BCELoss,
        optimizer=torch.optim.Adam,
    )
