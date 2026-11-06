
import torch
import torch.nn as nn


class DigitsClassifier(nn.Module):

    def __init__(self, n_layers=1, hidden=64, dropout=0.0):
        super().__init__()

        if n_layers not in (1, 2, 3):
            raise ValueError("n_layers должен быть 1, 2 или 3")

        if hidden < 1:
            raise ValueError("hidden должен быть положительным")

        if not 0 <= dropout <= 0.5:
            raise ValueError("dropout должен быть от 0 до 0.5")

        layers = []
        input_size = 64

        for _ in range(n_layers):
            layers.append(nn.Linear(input_size, hidden))
            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            input_size = hidden

        layers.append(nn.Linear(input_size, 10))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
