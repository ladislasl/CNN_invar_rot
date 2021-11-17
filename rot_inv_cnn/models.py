import torch
from torch import nn as nn
from torch.nn import functional as F

from code.rot_inv_cnn.layers import PolarConvNd


class Baseline(nn.Module):
    def __init__(self, input_width, n_filter=32, polar=False, n_input_channels=1):
        super(Baseline, self).__init__()
        conv_op = nn.Conv2d if not polar else PolarConvNd

        # compute fc input shape
        width_fc = (input_width - 2 - 2) // 2
        n_neurons_fc = width_fc * width_fc * (2 * n_filter)

        self.conv1 = conv_op(n_input_channels, n_filter, 3, 1)
        self.conv2 = conv_op(n_filter, 2*n_filter, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(n_neurons_fc, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class PolarBaseline(nn.Module):
    def __init__(self):
        super(PolarBaseline, self).__init__()
        self.conv1 = PolarConvNd(1, 32, 3, 1)
        self.conv2 = PolarConvNd(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class BaselineSmall(nn.Module):
    def __init__(self):
        super(BaselineSmall, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, 1)
        self.conv2 = nn.Conv2d(10, 20, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(2880, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
