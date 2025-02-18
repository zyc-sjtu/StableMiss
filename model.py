# -*- coding=utf-8 -*-


import math
import numpy as np
import torch
import torch.nn as nn


class MLPPredictor(nn.Module):
    def __init__(self, num_feature):
        super(MLPPredictor, self).__init__()
        self.linear_module = nn.Sequential(
            nn.Linear(in_features=num_feature + 1, out_features=num_feature + 1, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=num_feature + 1, out_features=num_feature + 1, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=num_feature + 1, out_features=num_feature + 1, bias=False)
        )
    def forward(self, x, m):
        x = torch.cat((torch.ones(x.size(0), 1), x), dim=1)
        m = torch.cat((torch.ones(m.size(0), 1), m), dim=1)
        alpha = self.linear_module(m)
        incomplete_x = torch.mul(x, m)
        y = torch.mul(alpha, incomplete_x).sum(dim=1)
        return y


class SampleReweighter(nn.Module):
    def __init__(self, num_sample):
        super(SampleReweighter, self).__init__()
        self.weight = nn.parameter.Parameter(torch.empty(num_sample).float())
        self.softmax = nn.Softmax()
        nn.init.normal_(self.weight)
    def forward(self):
        return self.softmax(self.weight)