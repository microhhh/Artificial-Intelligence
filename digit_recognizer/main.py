# coding: utf-8

import sys

sys.path.append('..')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import *
from digit_recognizer.mnist import MNIST, RandomShift
from digit_recognizer.net_structure import *

net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.003)
loss_func = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    net = net.cuda()
    loss_func = loss_func.cuda()

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


def train(epoch=1):
    net.train()
    exp_lr_scheduler.step()

    for iter, (data, label) in enumerate(train_loader):
        data, label = Variable(data), Variable(label)

        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()

        optimizer.zero_grad()
        output = net(data)
        loss = loss_func(output, label)
        loss.backward()
        optimizer.step()

        if (iter + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]  Loss: {:.6f}'.format(epoch, (iter + 1) * len(data),
                                                                           len(train_loader.dataset),
                                                                           100.0 * (iter + 1) / len(train_loader),
                                                                           loss.item()))


def evaluate(data_loader):
    net.eval()
    loss = 0
    correct = 0

    for data, label in data_loader:
        data, label = Variable(data, volatile=True), Variable(label)
        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()

        output = net(data)
        loss += F.cross_entropy(output, label, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()

    loss /= len(data_loader.dataset)

    print('  Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(loss, correct, len(data_loader.dataset),
                                                                       100. * correct / len(data_loader.dataset)))


def prediciton(data_loader):
    net.eval()
    test_pred = torch.LongTensor()

    for i, data in enumerate(data_loader):
        data = Variable(data, volatile=True)
        if torch.cuda.is_available():
            data = data.cuda()

        output = net(data)

        pred = output.cpu().data.max(1, keepdim=True)[1]
        test_pred = torch.cat((test_pred, pred), dim=0)
    return test_pred


if __name__ == '__main__':
    batch_size = 64
    n_pixels = 784

    train_dataset = MNIST('./data/train.csv', n_pixels=n_pixels, transform=transforms.Compose(
        [transforms.ToPILImage(), RandomRotation(degrees=20), RandomShift(shift=3), transforms.ToTensor(),
         transforms.Normalize(mean=(0.5,), std=(0.5,))]))
    test_dataset = MNIST('./data/test.csv', n_pixels=n_pixels)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    n_epochs = 50
    for epoch in range(n_epochs):
        train(epoch)
        evaluate(train_loader)

    test_prediciton = prediciton(test_loader)
    pred_df = pd.DataFrame(np.c_[np.arange(1, len(test_dataset) + 1)[:, None], test_prediciton.numpy()],
                           columns=['ImageId', 'Label'])
    pred_df.head()
    pred_df.to_csv('submission.csv', index=False)
