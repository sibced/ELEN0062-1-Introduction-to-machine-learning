import torch
from torch import Tensor, nn, softmax
from torch.nn.functional import sigmoid, dropout, tanh

class Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.ff1 = nn.Linear(47, 500)
        self.ff2 = nn.Linear(500, 22)

    def __call__(self, input: Tensor) -> Tensor:

        x = self.ff1(input)
        x = sigmoid(x)
        x = dropout(x, .3, self.training)

        x = self.ff2(x)
        x = sigmoid(x)
        
        # x = softmax(x, 1) -> done by crossentropy
        return x

class Deeper(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.ff1 = nn.Linear(47, 1000)
        self.bn1 = nn.BatchNorm1d(1000)

        self.ff2 = nn.Linear(1000, 500)
        self.bn2 = nn.BatchNorm1d(500)
        
        self.ff3 = nn.Linear(500, 100)
        self.bn3 = nn.BatchNorm1d(100)
        
        self.ff4 = nn.Linear(100, 50)
        self.bn4 = nn.BatchNorm1d(50)
        
        self.ff5 = nn.Linear(50, 22)

        # reused
        self.act = nn.LeakyReLU(.01)


    def __call__(self, input: Tensor) -> Tensor:

        x = self.ff1(input)
        x = self.bn1(x)
        x = tanh(x)
        x = dropout(x, .4, self.training)

        x = self.ff2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = dropout(x, .4, self.training)

        x = self.ff3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = dropout(x, .4, self.training)

        x = self.ff4(x)
        x = self.bn4(x)
        x = self.act(x)
        x = dropout(x, .4, self.training)

        x = self.ff5(x)
        # x = softmax(x, 1)  # <- the cross entropy loss already does it
        return x