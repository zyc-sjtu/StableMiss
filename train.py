# -*- coding=utf-8 -*-


import json
import numpy as np
import torch
from dataset import SyntheticDataset
from torch.utils.data import DataLoader
from model import MLPPredictor
import torch.optim as optim
from torch.autograd import grad
from metric import weighted_mean_squared_error, root_mean_square_error

NUM_FEATURE = 50

BATCH_SIZE = 64
NUM_EPOCH = 1000

TRAIN_DATA_PATH = ''
TRAIN_MASK_PATH = ''
WEIGHT_PATH = ''
MODEL_PATH = ''

if __name__ == '__main__':
    train_set = SyntheticDataset(TRAIN_DATA_PATH, TRAIN_MASK_PATH, WEIGHT_PATH)
    train_set_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    model = MLPPredictor(int(NUM_FEATURE))
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    train_loss, train_error = [], []
    for epoch in range(NUM_EPOCH):
        print ('Epoch ' + str(epoch + 1))
        epoch_loss, epoch_error = 0, 0
        for x, y, m, w in train_set_loader:
            y_hat = model(x, m)
            optimizer.zero_grad()
            loss = weighted_mean_squared_error(y, y_hat, w)
            epoch_loss += loss
            loss.backward()
            optimizer.step()
            rmse = root_mean_square_error(y, y_hat)
            epoch_error += rmse
        epoch_loss /= len(train_set_loader)
        epoch_error /= len(train_set_loader)
        train_loss.append(epoch_loss.item())
        train_error.append(epoch_error.item())
        print ('Train loss: ' + str(train_loss[-1]))
        print ('Train error: ' + str(train_error[-1]))
        if (epoch + 1) % 5 == 0:
            model_num = int((epoch + 1) / 5)
            torch.save(model.state_dict(), MODEL_PATH + 'model_' + str(model_num) + '.pth')
    log = {'train_loss': train_loss, 'train_error': train_error}
    json.dump(log, open(MODEL_PATH + 'log.json', 'w'))