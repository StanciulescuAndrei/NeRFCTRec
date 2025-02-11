import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torchvision
import numpy as np

class NHGrid(nn.Module):
    def __init__(self, numInputFeatures, numGridFeatures, gridLevels, hashSize):
        super(NHGrid, self).__init__()
        self.numInputFeatures = numInputFeatures
        self.numGridFeatures = numGridFeatures
        self.gridLevels = gridLevels
        self.hashSize = hashSize

        self.dimIncrease = 1.6
        self.primes = torch.tensor([73856093, 19349663]).cuda()

        self.hashFeatures = nn.ModuleList([
            nn.Embedding(self.hashSize, self.numGridFeatures) for _ in range(self.gridLevels)
        ])
        self.block1 = nn.Sequential(
            nn.Linear(self.gridLevels * self.numGridFeatures, 32), nn.LeakyReLU(),
            nn.Linear(32, 16), nn.LeakyReLU(),
            nn.Linear(16, 8), nn.LeakyReLU(),
            nn.Linear(8, 1), nn.LeakyReLU(),
        )

        self.hashSpacing = []
        for i in range(self.gridLevels):
            self.hashSpacing.append(256 // np.power(self.dimIncrease, self.gridLevels - i - 1))
    
    def positionalEncoding(self, x):
        out = []
        for i in range(self.gridLevels):
            spacing = self.hashSpacing[i]
            discrete_coords = (x * spacing).floor().int()
            hash_index = (discrete_coords * self.primes).sum(dim=-1) % self.hashSize
            features = self.hashFeatures[i](hash_index)
            out.append(features)
        return torch.cat(out, dim=1)

    def forward(self, x, trueRange = False):
        x = x * 0.5 + 0.5
        hashFeatures = self.positionalEncoding(x)
        out = self.block1(hashFeatures)
        if trueRange:
            out = nn.ReLU().forward(out)
        else:
            out = nn.LeakyReLU().forward(out)
        return out