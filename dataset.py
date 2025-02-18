# -*- coding=utf-8 -*-


import json
import math
from scipy.optimize import fsolve
import pandas as pd
import numpy as np
from numpy.random import choice, randint, rand, uniform, randn, normal, multivariate_normal
import torch
from torch.utils.data import Dataset


class GaussianIndependentDataGenerator:
    def __init__(self, num_feature, path):
        self.x_mean = normal(loc=0, scale=1, size=num_feature)
        temp = normal(loc=0, scale=1, size=(num_feature, int(num_feature * 0.7)))
        self.x_cov = np.matmul(temp, np.transpose(temp)) + np.diag(uniform(low=0.01, high=0.1, size=num_feature))
        self.x_cov = np.diag(np.diag(self.x_cov))
        self.alpha = np.repeat(1.0, num_feature + 1)
        alpha_std = math.sqrt(np.dot(np.dot(self.alpha[1:], self.x_cov), self.alpha[1:]))
        self.alpha[1:] += normal(loc=self.alpha[1:], scale=alpha_std, size=num_feature)
        params = {'x_mean': self.x_mean.tolist(), 'x_cov': self.x_cov.tolist(), 'alpha': self.alpha.tolist()}
        json.dump(params, open(path, 'w'))
    def generate(self, num_sample, path, snr=10):
        x = multivariate_normal(mean=self.x_mean, cov=self.x_cov, size=num_sample)
        y_mean = x.dot(self.alpha[1:]) + self.alpha[0]
        y_std = math.sqrt(np.var(y_mean) / snr)
        y = normal(loc=y_mean, scale=y_std, size=num_sample)
        data = {'x': x.tolist(), 'y': y.tolist()}
        json.dump(data, open(path, 'w'))
        return x, y


class GaussianDataGenerator:
    def __init__(self, num_feature, path):
        self.x_mean = normal(loc=0, scale=1, size=num_feature)
        temp = normal(loc=0, scale=1, size=(num_feature, int(num_feature * 0.7)))
        self.x_cov = np.matmul(temp, np.transpose(temp)) + np.diag(uniform(low=0.01, high=0.1, size=num_feature))
        self.alpha = np.repeat(1.0, num_feature + 1)
        alpha_std = math.sqrt(np.dot(np.dot(self.alpha[1:], self.x_cov), self.alpha[1:]))
        self.alpha[1:] += normal(loc=self.alpha[1:], scale=alpha_std, size=num_feature)
        params = {'x_mean': self.x_mean.tolist(), 'x_cov': self.x_cov.tolist(), 'alpha': self.alpha.tolist()}
        json.dump(params, open(path, 'w'))
    def generate(self, num_sample, path, snr=10):
        x = multivariate_normal(mean=self.x_mean, cov=self.x_cov, size=num_sample)
        y_mean = x.dot(self.alpha[1:]) + self.alpha[0]
        y_std = math.sqrt(np.var(y_mean) / snr)
        y = normal(loc=y_mean, scale=y_std, size=num_sample)
        data = {'x': x.tolist(), 'y': y.tolist()}
        json.dump(data, open(path, 'w'))
        return x, y


class GaussianMixtureDataGenerator:
    def __init__(self, num_component, num_feature, path):
        self.pi = rand(num_component)
        self.pi /= np.sum(self.pi)
        self.x_mean, self.x_cov = [], []
        for _ in range(num_component):
            x_mean = normal(loc=0, scale=1, size=num_feature) / 2
            temp = normal(loc=0, scale=1, size=(num_feature, int(num_feature * 0.7)))
            x_cov = np.matmul(temp, temp.T) + np.diag(uniform(low=0.01, high=0.1, size=num_feature))
            self.x_mean.append(x_mean)
            self.x_cov.append(x_cov)
        temp = normal(loc=0, scale=1, size=(num_feature, int(num_feature * 0.7)))
        temp_cov = np.matmul(temp, temp.T) + np.diag(uniform(low=0.01, high=0.1, size=num_feature))
        self.alpha = np.repeat(1.0, num_feature + 1)
        alpha_std = math.sqrt(np.dot(np.dot(self.alpha[1:], temp_cov), self.alpha[1:]))
        self.alpha[1:] += normal(loc=self.alpha[1:], scale=alpha_std, size=num_feature)
        params = {'pi': self.pi.tolist(), 'x_mean': np.array(self.x_mean).tolist(), 'x_cov': np.array(self.x_cov).tolist(), 'alpha': self.alpha.tolist()}
        json.dump(params, open(path, 'w'))
    def generate(self, num_sample, path, snr=10):
        sample_component = rand(num_sample)
        x_list = []
        for i in range(self.pi.shape[0]):
            num_component_sample = np.count_nonzero((sample_component >= np.sum(self.pi[:i])) * (sample_component < np.sum(self.pi[:i + 1])))
            x_component = multivariate_normal(mean=self.x_mean[i], cov=self.x_cov[i], size=num_component_sample)
            x_list.append(x_component)
        x = np.concatenate(tuple(x_list), axis=0)
        y_mean = x.dot(self.alpha[1:]) + self.alpha[0]
        y_std = math.sqrt(np.var(y_mean) / snr)
        y = normal(loc=y_mean, scale=y_std, size=num_sample)
        data = {'x': x.tolist(), 'y': y.tolist()}
        json.dump(data, open(path, 'w'))
        return x, y


class IndependentMaskGenerator:
    def __init__(self, missing_rate, path):
        self.missing_rate = missing_rate
        params = {'missing_rate': self.missing_rate}
        json.dump(params, open(path, 'w'))
    def generate(self, num_sample, num_feature, path):
        missing_rate_proportion = [0.025 for _ in range(9)]
        missing_rate_proportion[int(self.missing_rate * 10) - 1] = 0.8
        sample_missing_rate = rand(num_sample)
        m_list = []
        for i in range(9):
            num_missing_rate_sample = np.count_nonzero((sample_missing_rate >= np.sum(missing_rate_proportion[:i])) * (sample_missing_rate < np.sum(missing_rate_proportion[:i + 1])))
            m_missing_rate = rand(num_missing_rate_sample, num_feature)
            m_missing_rate[m_missing_rate < ((i + 1) * 0.1)] = 0
            m_missing_rate[m_missing_rate >= ((i + 1) * 0.1)] = 1
            m_list.append(m_missing_rate)
        m = np.concatenate(tuple(m_list), axis=0)
        mask = {'m': m.tolist()}
        json.dump(mask, open(path, 'w'))
        return m


class DependentMaskGenerator:
    def __init__(self, missing_rate, path):
        self.missing_rate = missing_rate
        params = {'missing_rate': self.missing_rate}
        json.dump(params, open(path, 'w'))
    def generate(self, num_sample, num_feature, path):
        missing_rate_proportion = [0.025 for _ in range(9)]
        missing_rate_proportion[int(self.missing_rate * 10) - 1] = 0.8
        sample_missing_rate = rand(num_sample)
        m_list = []
        for i in range(9):
            num_missing_rate_sample = np.count_nonzero((sample_missing_rate >= np.sum(missing_rate_proportion[:i])) * (sample_missing_rate < np.sum(missing_rate_proportion[:i + 1])))
            m_missing_rate = np.ones((num_missing_rate_sample, num_feature))
            missing_length = int(num_feature * (i + 1) * 0.1)
            for j in range(num_missing_rate_sample):
                start_index = randint(low=0, high=num_feature)
                if start_index + missing_length <= num_feature:
                    m_missing_rate[j][start_index : start_index + missing_length] = 0
                else:
                    m_missing_rate[j][start_index:] = 0
                    m_missing_rate[j][:start_index + missing_length - num_feature] = 0
            m_list.append(m_missing_rate)
        m = np.concatenate(tuple(m_list), axis=0)
        mask = {'m': m.tolist()}
        json.dump(mask, open(path, 'w'))
        return m


class MARMaskGenerator:
    def __init__(self, missing_rate, path):
        self.missing_rate = missing_rate
        params = {'missing_rate': self.missing_rate}
        json.dump(params, open(path, 'w'))
    def generate(self, x, path):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        num_sample, num_feature = x.shape
        num_observed = int(num_feature * 0.1)
        num_incomplete = num_feature - num_observed
        index_observed = choice(num_feature, num_observed, replace=False)
        index_incomplete = np.array([i for i in range(num_feature) if i not in index_observed])
        x_mean = np.mean(x, axis=0)
        temp = np.dot(np.transpose(x - x_mean), (x - x_mean)) / num_sample
        temp_observed = temp[np.ix_(index_observed, index_observed)]
        coefficient = randn(num_observed, num_incomplete)
        v = np.array([np.dot(np.dot(coefficient[:, i], temp_observed), coefficient[:, i]) for i in range(num_incomplete)])
        steepness = uniform(low=0.1, high=0.5, size=num_incomplete)
        coefficient /= steepness * np.sqrt(v)
        missing_rate_proportion = [0.025 for _ in range(9)]
        missing_rate_proportion[int(self.missing_rate * 10) - 1] = 0.8
        sample_missing_rate = rand(num_sample)
        m_list = []
        num_sample_cumulative = 0
        for i in range(9):
            num_missing_rate_sample = np.count_nonzero((sample_missing_rate >= np.sum(missing_rate_proportion[:i])) * (sample_missing_rate < np.sum(missing_rate_proportion[:i + 1])))
            interception = np.zeros((num_incomplete,))
            for j in range(num_incomplete):
                w = coefficient[:, j]
                def f(b):
                    s = sigmoid(np.dot(x[:, index_observed], w) + b) - ((i + 1) * 0.1)
                    return s.mean()
                result = fsolve(f, x0=0)
                interception[j] = result[0]
            m_missing_rate = np.ones((num_missing_rate_sample, num_feature))
            threshold = sigmoid(np.dot(x[num_sample_cumulative: num_sample_cumulative + num_missing_rate_sample, index_observed], coefficient) + interception)
            num_sample_cumulative += num_missing_rate_sample
            temp = rand(num_missing_rate_sample, num_incomplete)
            m_missing_rate[:, index_incomplete] = temp >= threshold
            m_list.append(m_missing_rate)
        m = np.concatenate(tuple(m_list), axis=0)
        mask = {'m': m.tolist()}
        json.dump(mask, open(path, 'w'))
        return m


class SyntheticDataset(Dataset):
    def __init__(self, data_path, mask_path, weight_path=None):
        data = json.load(open(data_path, 'r'))
        mask = json.load(open(mask_path, 'r'))
        self.x = torch.tensor(data['x']).float()
        self.y = torch.tensor(data['y']).float()
        self.m = torch.tensor(mask['m']).float()
        if weight_path == None:
            self.w = torch.ones(self.x.size(0)).float()
        else:
            self.w = torch.load(weight_path).float()
    def __getitem__(self, index):
        return self.x[index], self.y[index], self.m[index], self.w[index]
    def __len__(self):
        return self.x.size(0)