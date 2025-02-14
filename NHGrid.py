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

        self.dimIncrease = 1.3
        self.factors = torch.tensor([1, 64]).cuda()

        self.hashFeatures = nn.ModuleList([
            nn.Embedding(self.hashSize, self.numGridFeatures) for _ in range(self.gridLevels)
        ])
        self.block1 = nn.Sequential(
            nn.Linear(self.gridLevels * self.numGridFeatures, 64), nn.LeakyReLU(),
            nn.Linear(64, 32), nn.LeakyReLU(),
            nn.Linear(32, 16), nn.LeakyReLU(),
            nn.Linear(16, 8), nn.LeakyReLU(),
            nn.Linear(8, 1),
        )

        self.hashSpacing = []
        for i in range(self.gridLevels):
            self.hashSpacing.append(64 // np.power(self.dimIncrease, self.gridLevels - i - 1))
    
    def positionalEncoding(self, x):
        out = []
        for i in range(self.gridLevels):
            spacing = self.hashSpacing[i]
            discrete_coords = torch.zeros([x.shape[0], x.shape[1], 4], device="cuda")
            discrete_coords[:, :, 0] = (x * spacing).floor().int()
            discrete_coords[:, :, 1] = (x * spacing).floor().int() + torch.tensor([0, 1], device="cuda")
            discrete_coords[:, :, 2] = (x * spacing).floor().int() + torch.tensor([1, 0], device="cuda")
            discrete_coords[:, :, 3] = (x * spacing).floor().int() + torch.tensor([1, 1], device="cuda")
            weights = torch.norm(x[:, :, None] * spacing - discrete_coords, dim=1)
            weights = torch.nn.functional.normalize(1.42 - weights, p=1.0, dim=1)

            hash_index = (discrete_coords * self.factors[:, None]).sum(dim=-2) % self.hashSize
            features = self.hashFeatures[i](hash_index.long())
            interp_features = (features * weights[:, :, None]).sum(dim=-2)
            out.append(interp_features)

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