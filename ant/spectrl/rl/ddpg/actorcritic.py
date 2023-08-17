"""
Definitions for Actor and Critic
Author: Sameera Lanka
Website: https://sameera-lanka.com
Modified for DIRL
"""

import torch
import torch.nn as nn
import numpy as np


WFINAL = 0.003


def fanin_init(size, fanin=None):
    """Utility function for initializing actor and critic"""
    fanin = fanin or size[1]
    w = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-w, w)


class Actor(nn.Module):
    """Defines actor network"""

    def __init__(self, stateDim, actionDim, hiddenDim, actionBound):
        super(Actor, self).__init__()

        self.actionBound = torch.Tensor(actionBound)

        # normalization layer
        self.norm0 = nn.BatchNorm1d(stateDim)

        # input layer
        self.input_layer = nn.Linear(stateDim, hiddenDim)

        # hidden layer
        self.hidden_layer = nn.Linear(hiddenDim, hiddenDim)

        # output layer
        self.output_layer = nn.Linear(hiddenDim, actionDim)

        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
        self.use_gpu = False

    def forward(self, state):
        state = self.norm0(state)
        h1 = self.ReLU(self.input_layer(state))
        h2 = self.ReLU(self.hidden_layer(h1))
        action = self.actionBound * self.Tanh(self.output_layer(h2))
        return action

    def get_action(self, state, using_tensors=False):

        # set eval mode
        self.eval()

        # convert to tensor
        if not using_tensors:
            state = torch.Tensor(state)
            if self.use_gpu:
                state = state.cuda()

        # batch of one
        state = state.view(1, -1)
        with torch.no_grad():
            action = self.forward(state)[0]

        # switch to train mode
        self.train()

        # return tensor or numpy array
        if using_tensors:
            return action
        else:
            return action.cpu().numpy()

    def set_use_gpu(self):
        self.use_gpu = True
        self.actionBound = self.actionBound.cuda()

    def set_use_cpu(self):
        self.use_gpu = False
        self.actionBound = self.actionBound.cpu()
        return self.cpu()


class Critic(nn.Module):
    """Defines critic network"""

    def __init__(self, stateDim, actionDim, hiddenDim):
        super(Critic, self).__init__()

        # normalization layer
        self.norm1 = nn.BatchNorm1d(stateDim)

        # input layer
        self.input_layer = nn.Linear(stateDim + actionDim, hiddenDim)

        # hidden layer
        self.hidden_layer = nn.Linear(hiddenDim, hiddenDim)

        # output layer
        self.output_layer = nn.Linear(hiddenDim, 1)

        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()

    def forward(self, state, action):
        state = self.norm1(state)
        h1 = self.ReLU(self.input_layer(torch.cat([state, action], dim=1)))
        h2 = self.ReLU(self.hidden_layer(h1))
        Qval = self.output_layer(h2)
        return Qval
