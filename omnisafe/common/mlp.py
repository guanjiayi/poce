import torch
import numpy as np
from torch import nn
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union, Callable, Optional, Sequence


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MLP(nn.Module):
    r"""
        Multi-layer Perceptron
        Inputs:
            in_features : int, features numbers of the input
            out_features : int, features numbers of the output
            hidden_features : int, features numbers of the hidden layers
            hidden_layers : int, numbers of the hidden layers
            norm : str, normalization method between hidden layers, default : None
            hidden_activation : str, activation function used in hidden layers, default : 'leakyrelu'
            output_activation : str, activation function used in output layer, default : 'identity'
    """

    ACTIVATION_CREATORS = {
        'relu' : lambda: nn.ReLU(inplace=True),
        'elu' : lambda: nn.ELU(),
        'leakyrelu' : lambda: nn.LeakyReLU(negative_slope=0.1, inplace=True),
        'tanh' : lambda: nn.Tanh(),
        'sigmoid' : lambda: nn.Sigmoid(),
        'identity' : lambda: nn.Identity(),
        'gelu' : lambda: nn.GELU(),
        'swish' : lambda: Swish(),
    }

    def __init__(self, 
                 in_features : int, 
                 out_features : int, 
                 hidden_features : int, 
                 hidden_layers : int, 
                 norm : str = None, 
                 hidden_activation : str = 'leakyrelu', 
                 output_activation : str = 'identity'):
        super(MLP, self).__init__()

        hidden_activation_creator = self.ACTIVATION_CREATORS[hidden_activation]
        output_activation_creator = self.ACTIVATION_CREATORS[output_activation]

        if hidden_layers == 0:
            self.net = nn.Sequential(
                nn.Linear(in_features, out_features),
                output_activation_creator(out_features)
            )
        else:
            net = []
            for i in range(hidden_layers):
                net.append(nn.Linear(in_features if i == 0 else hidden_features, hidden_features))
                if norm:
                    if norm == 'ln':
                        net.append(nn.LayerNorm(hidden_features))
                    elif norm == 'bn':
                        net.append(nn.BatchNorm1d(hidden_features))
                    else:
                        raise NotImplementedError(f'{norm} does not supported!')
                net.append(hidden_activation_creator())
            net.append(nn.Linear(hidden_features, out_features))
            net.append(output_activation_creator())
            self.net = nn.Sequential(*net)

    def forward(self, x):
        r"""forward method of MLP only assume the last dim of x matches `in_features`"""
        head_shape = x.shape[:-1]
        x = x.view(-1, x.shape[-1])
        out = self.net(x)
        out = out.view(*head_shape, out.shape[-1])
        return out