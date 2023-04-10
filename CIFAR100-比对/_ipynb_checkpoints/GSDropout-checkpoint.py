import torch
import numpy as np


class GaussianDropout:
    def __init__(self, dropout_prob):
        self.dropout_prob = dropout_prob
        self.mask = None

    def forward(self, x, train=True):
        if train:
            self.mask = torch.normal(mean=torch.ones_like(x), std=self.dropout_prob)
            out = x * self.mask
        else:
            out = x
        return out

    def backward(self, dout):
        dx = dout * self.mask
        return dx
