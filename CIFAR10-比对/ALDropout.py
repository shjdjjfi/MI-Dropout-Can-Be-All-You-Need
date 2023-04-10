import torch
import numpy as np


class AlphaDropout:
    def __init__(self, dropout_prob, alpha=1.0, scale=1.0):
        self.dropout_prob = dropout_prob
        self.alpha = alpha
        self.scale = scale
        self.mask = None

    def forward(self, x, train=True):
        if train:
            alpha = self.alpha
            if alpha == 1.0:
                alpha = 1.5  # default value for alpha in AlphaDropout
            self.mask = torch.randn_like(x) * alpha + 1
            self.mask = self.mask.clamp(min=0) / (self.scale + alpha ** 2 * self.dropout_prob * self.scale)
            out = x * self.mask
        else:
            out = x
        return out

    def backward(self, dout):
        dx = dout * self.mask
        return dx
