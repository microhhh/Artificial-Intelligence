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
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import *
from torchvision.utils import make_grid
import math
from PIL import Image
import matplotlib.pyplot as plt

from digit_recognizer.mnist import MNIST_data, RandomShift


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10),
        )

        for m in self.features.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.classifier.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.003)
criterion = nn.CrossEntropyLoss()

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()


def train(epoch):
    model.train()
    exp_lr_scheduler.step()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))


def evaluate(data_loader):
    model.eval()
    loss = 0
    correct = 0

    for data, target in data_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        output = model(data)

        loss += F.cross_entropy(output, target, size_average=False).item()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    loss /= len(data_loader.dataset)

    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))


def prediciton(data_loader):
    model.eval()
    test_pred = torch.LongTensor()

    for i, data in enumerate(data_loader):
        data = Variable(data, volatile=True)
        if torch.cuda.is_available():
            data = data.cuda()

        output = model(data)

        pred = output.cpu().data.max(1, keepdim=True)[1]
        test_pred = torch.cat((test_pred, pred), dim=0)

    return test_pred


if __name__ == '__main__':
    train_df = pd.read_csv('./data/train.csv')
    n_train = len(train_df)
    n_pixels = len(train_df.columns) - 1
    n_class = len(set(train_df['label']))

    print('Number of training samples: {0}'.format(n_train))
    print('Number of training pixels: {0}'.format(n_pixels))
    print('Number of classes: {0}'.format(n_class))

    test_df = pd.read_csv('./data/test.csv')
    n_test = len(test_df)
    n_pixels = len(test_df.columns)

    print('Number of train samples: {0}'.format(n_test))
    print('Number of test pixels: {0}'.format(n_pixels))

    random_sel = np.random.randint(n_train, size=8)

    grid = make_grid(torch.Tensor((train_df.iloc[random_sel, 1:].values / 255.).reshape((-1, 28, 28))).unsqueeze(1),
                     nrow=8)
    plt.rcParams['figure.figsize'] = (16, 2)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.axis('off')
    print(*list(train_df.iloc[random_sel, 0].values), sep=', ')
    plt.show()

    plt.rcParams['figure.figsize'] = (8, 5)
    plt.bar(train_df['label'].value_counts().index, train_df['label'].value_counts())
    plt.xticks(np.arange(n_class))
    plt.xlabel('Class', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.grid(True, axis='y')
    plt.show()

    batch_size = 64

    train_dataset = MNIST_data('./data/train.csv', n_pixels=n_pixels, transform=transforms.Compose(
        [transforms.ToPILImage(), RandomRotation(degrees=20), RandomShift(3),
         transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]))
    test_dataset = MNIST_data('./data/test.csv', n_pixels=n_pixels)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size, shuffle=False)

    rotate = RandomRotation(20)
    shift = RandomShift(3)
    composed = transforms.Compose([RandomRotation(20), RandomShift(3)])

    # Apply each of the above transforms on sample.
    fig = plt.figure()
    sample = transforms.ToPILImage()(train_df.iloc[65, 1:].values.reshape((28, 28)).astype(np.uint8)[:, :, None])
    for i, tsfrm in enumerate([rotate, shift, composed]):
        transformed_sample = tsfrm(sample)

        ax = plt.subplot(1, 3, i + 1)
        plt.tight_layout()
        ax.set_title(type(tsfrm).__name__)
        ax.imshow(np.reshape(np.array(list(transformed_sample.getdata())), (-1, 28)), cmap='gray')

    plt.show()

    n_epochs = 50
    for epoch in range(n_epochs):
        train(epoch)
        evaluate(train_loader)
    test_pred = prediciton(test_loader)
    out_df = pd.DataFrame(np.c_[np.arange(1, len(test_dataset) + 1)[:, None], test_pred.numpy()],
                          columns=['ImageId', 'Label'])
    out_df.head()
    out_df.to_csv('submission.csv', index=False)
