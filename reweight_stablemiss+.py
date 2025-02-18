# -*- coding=utf-8 -*-


import json
import numpy as np
import torch
from model import SampleReweighter
import torch.optim as optim
from torch.autograd import grad
from metric import intra_feature_correlation, intra_mask_correlation, inter_correlation, w_distance


NUM_SAMPLE = 16384

NUM_EPOCH = 20000
LAMBDA = 10

TRAIN_DATA_PATH = ''
TRAIN_MASK_PATH = ''
IND_DATA_PATH = ''
IND_MASK_PATH = ''
MODEL_PATH = ''


if __name__ == '__main__':
    data = json.load(open(TRAIN_DATA_PATH, 'r'))
    mask = json.load(open(TRAIN_MASK_PATH, 'r'))
    data_ind = json.load(open(IND_DATA_PATH, 'r'))
    mask_ind = json.load(open(IND_MASK_PATH, 'r'))
    x = torch.tensor(data['x']).float()
    m = torch.tensor(mask['m']).float()
    x_ind = torch.tensor(data_ind['x']).float()
    m_ind = torch.tensor(mask_ind['m']).float()
    model = SampleReweighter(NUM_SAMPLE)
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    train_loss, train_error = [], []
    for epoch in range(NUM_EPOCH):
        print ('Epoch ' + str(epoch + 1))
        optimizer.zero_grad()
        w = model()
        loss = w_distance(x, m, x_ind, m_ind, w) + intra_mask_correlation(m, w) + inter_correlation(x, m, w)
        loss += LAMBDA * (w.std() / w.mean())
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        train_error.append(loss.item())
        print ('Train loss: ' + str(train_loss[-1]))
        print ('Train error: ' + str(train_error[-1]))
    torch.save(model.state_dict(), MODEL_PATH + 'model.pth')
    log = {'train_loss': train_loss, 'train_error': train_error}
    json.dump(log, open(MODEL_PATH + 'log.json', 'w'))
    w = model()
    torch.save(w, MODEL_PATH + 'weight.pt')