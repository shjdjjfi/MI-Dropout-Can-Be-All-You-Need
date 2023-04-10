import argparse
import os
import torch
import numpy as np


class Dropout:
    def __init__(self, dropout_prob):
        self.dropout_prob = dropout_prob
        self.mask = None

    def forward(self, x, train=True):
        if train:
            self.mask = np.random.binomial(1, 1 - self.dropout_prob, size=x.shape).astype(np.float32) / (
                    1 - self.dropout_prob)
            if torch.cuda.is_available():
                self.mask = torch.from_numpy(self.mask).to(x.device)
            out = x * self.mask
        else:
            out = x
        return out

    def backward(self, dout):
        dx = dout * self.mask
        return dx
