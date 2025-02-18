# -*- coding=utf-8 -*-


import numpy as np
import ot
import torch
import torch.nn as nn
from torch.autograd import Variable


def weighted_mean_squared_error(y, y_hat, w):
    weighted_mean_squared_error = (torch.square(y_hat - y) * w).sum()
    return weighted_mean_squared_error


def root_mean_square_error(y, y_hat):
    root_mean_square_error = torch.sqrt(torch.square(y_hat - y).mean())
    return root_mean_square_error


def weighted_covariance(x, m, w):
    weighted_x_cov = torch.cov(torch.transpose(x, 1, 0), aweights=w)
    weighted_m_cov = torch.cov(torch.transpose(m, 1, 0), aweights=w)
    temp = (torch.ones(x.size(1), x.size(1)) - torch.diag(torch.ones(x.size(1)))).float()
    weighted_x_covariance = torch.linalg.matrix_norm(weighted_x_cov * temp)
    weighted_m_covariance = torch.linalg.matrix_norm(weighted_m_cov * temp)
    weighted_covariance = weighted_x_covariance + weighted_m_covariance
    return weighted_covariance


def random_fourier_features(x, w=None, b=None, num_f=None, sum=True, sigma=None, seed=None):
    if num_f is None:
        num_f = 1
    n = x.size(0)
    r = x.size(1)
    x = x.view(n, r, 1)
    c = x.size(2)
    if sigma is None or sigma == 0:
        sigma = 1
    if w is None:
        w = 1 / sigma * (torch.randn(size=(num_f, c)))
        b = 2 * np.pi * torch.rand(size=(r, num_f))
        b = b.repeat((n, 1, 1))
    Z = torch.sqrt(torch.tensor(2.0 / num_f))
    mid = torch.matmul(x, w.t())
    mid = mid + b
    mid -= mid.min(dim=1, keepdim=True)[0]
    mid /= mid.max(dim=1, keepdim=True)[0]
    mid *= np.pi / 2.0
    if sum:
        Z = Z * (torch.cos(mid) + torch.sin(mid))
    else:
        Z = Z * torch.cat((torch.cos(mid), torch.sin(mid)), dim=-1)
    return Z


def cov(x, w=None):
    if w is None:
        n = x.shape[0]
        cov = torch.matmul(x.t(), x) / n
        e = torch.mean(x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())
    else:
        w = w.view(-1, 1)
        cov = torch.matmul((w * x).t(), x)
        e = torch.sum(w * x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())
    return res


def intra_feature_correlation(x, m, weight):
    cfeaturecs = random_fourier_features(x, num_f=None, sum=sum)
    loss = Variable(torch.FloatTensor([0]))
    for i in range(cfeaturecs.size()[-1]):
        cfeaturec = cfeaturecs[:, :, i]
        cov1 = torch.mul(cov(cfeaturec, weight), m)
        cov_matrix = cov1 * cov1
        loss += torch.sum(cov_matrix) - torch.trace(cov_matrix)
    return loss


def intra_mask_correlation(m, weight):
    cfeaturecs = random_fourier_features(m, num_f=None, sum=sum)
    loss = Variable(torch.FloatTensor([0]))
    for i in range(cfeaturecs.size()[-1]):
        cfeaturec = cfeaturecs[:, :, i]
        cov1 = cov(cfeaturec, weight)
        cov_matrix = cov1 * cov1
        loss += torch.sum(cov_matrix) - torch.trace(cov_matrix)
    return loss


def inter_correlation(x, m , weight):
    cfeaturecs1 = random_fourier_features(x, num_f=None, sum=sum)
    cfeaturecs2 = random_fourier_features(m, num_f=None, sum=sum)
    loss = Variable(torch.FloatTensor([0]))
    for i in range(cfeaturecs1.size()[-1]):
        cfeaturec1 = cfeaturecs1[:, :, i]
        cfeaturec2 = cfeaturecs2[:, :, i]
        cov1 = torch.mul(cov(cfeaturec1, weight), m)
        cov2 = cov(cfeaturec2, weight)
        cov_matrix = cov1 * cov2
        loss += torch.sum(cov_matrix) - torch.trace(cov_matrix)
    return loss


def w_distance(x, m, x_ind, m_ind, weight):
    u = torch.mul(x, m)
    v = torch.mul(x_ind, m_ind)
    return ot.wasserstein_1d(u, v, u_weights=weight, v_weights=None)