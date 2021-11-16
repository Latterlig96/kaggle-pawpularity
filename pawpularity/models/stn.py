import torch
import torch.nn as nn
import torch.nn.functional as F


class StnLarge(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=10),
            nn.MaxPool2d(3, stride=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 10, kernel_size=8),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 12, kernel_size=8),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 14, kernel_size=8),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(14 * 25 * 25, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3 * 2)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor(
            [1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 14 * 25 * 25)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        x = self.stn(x)
        return x


class StnSmall(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=10),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 10, kernel_size=8),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 90 * 90, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3 * 2)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor(
            [1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 90 * 90)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        x = self.stn(x)
        return x
