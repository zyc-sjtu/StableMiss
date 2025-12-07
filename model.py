# -*- coding=utf-8 -*-


import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils


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


class ConvUNet(nn.Module):
    def __init__(self, in_channels=1, init_channels=32):
        super(ConvUNet, self).__init__()
        self.in_channels = in_channels
        self.init_channels = init_channels
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(init_channels, init_channels*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(init_channels*2),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(init_channels*2, init_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(init_channels, in_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x_down1 = self.down1(x)
        x_down2 = self.down2(x_down1)
        x_up1 = self.up1(x_down2)
        x_up2 = self.up2(x_up1)
        x_up2 = self.activation(x_up2)
        return x_up2


class ConvUNetWithMLP(nn.Module):
    def __init__(self, num_param, in_channels=1, init_channels=32):
        super(ConvUNetWithMLP, self).__init__()
        self.in_channels = in_channels
        self.init_channels = init_channels
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(init_channels, init_channels*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(init_channels*2),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(init_channels*2, init_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(init_channels, in_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.activation = nn.ReLU(inplace=True)
        self.mlp = nn.Sequential(
            nn.Linear(784, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, num_param)
        )
    def forward(self, x):
        x_down1 = self.down1(x)
        x_down2 = self.down2(x_down1)
        x_up1 = self.up1(x_down2)
        x_up2 = self.up2(x_up1)
        x_up2 = self.activation(x_up2)
        x_mlp_in = x_up2.reshape(x_up2.shape[0], -1)
        x_mlp_out = self.mlp(x_mlp_in)
        return x_mlp_out


class ConvPredictor(nn.Module):
    def __init__(self, x_dim, m_dim, g_hidden_dim, output_dim, f_hidden_dims):
        super(ConvPredictor, self).__init__()
        self.g_network = ConvUNet()
        num_param = sum(p.numel() for p in self.g_network.parameters() if p.requires_grad)
        self.phi_network = ConvUNetWithMLP(num_param)
    def forward(self, x, m):
        phi = self.phi_network(m)
        utils.vector_to_parameters(phi.reshape(-1), self.g_network.parameters())
        incomplete_x = torch.mul(x, m)
        y = self.g_network(incomplete_x)
        return y


class SampleReweighter(nn.Module):
    def __init__(self, num_sample):
        super(SampleReweighter, self).__init__()
        self.weight = nn.parameter.Parameter(torch.empty(num_sample).float())
        self.softmax = nn.Softmax()
        nn.init.normal_(self.weight)
    def forward(self):
        return self.softmax(self.weight)
