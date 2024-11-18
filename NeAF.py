import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NeAF(nn.Module):
    def __init__(self, numInputFeatures, encodingDegree):
        super(NeAF, self).__init__()
        self.numInputFeatures = numInputFeatures
        self.encodingDegree = encodingDegree
        self.block1 = nn.Sequential(
            nn.Linear(2 * encodingDegree * self.numInputFeatures + 2, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Linear(2 * encodingDegree * self.numInputFeatures + 2 + 256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.relu = nn.ReLU()

    @staticmethod
    def positionalEncoding(x, L):
        out = [x]
        for i in range(L):
            out.append(torch.sin(2 ** i * x))
            out.append(torch.cos(2 ** i * x))
        return torch.cat(out, dim=1)

    def forward(self, x):
        emb_x = self.positionalEncoding(x, self.encodingDegree)
        h = self.block1(emb_x)
        out = self.block2(torch.cat((h, emb_x), dim=1))
        density = self.relu(out)
        return density
    

def renderRays(neaf_model, geometryDescriptor, batchSize, numSamplePoints):
    # numSamplePoints x pixelCount x projectionCount

    samplePoints = []
    dt = []

    for desc in geometryDescriptor:
        for px in desc['pixels']:
            sp = np.array(desc['src'])
            ep = np.array(px)
            dt.append(np.linalg.norm(ep - sp))
            for t in np.linspace(0, 1, numSamplePoints):
                samplePoints.append(sp * (1.0 - t) + ep * t)
                

    samplePoints = torch.tensor(np.array(samplePoints), dtype=torch.float32, requires_grad=True).squeeze(1)
    densities = neaf_model(samplePoints)
    dt = torch.tensor(np.array(dt), dtype=torch.float32, requires_grad=True)

    densities = densities.view(batchSize, numSamplePoints)

    accum = torch.zeros(batchSize, dtype=torch.float32, requires_grad=True)
    T = torch.ones(batchSize, requires_grad=True)
    for sample in range(numSamplePoints):
        alpha = torch.exp(-densities[:, sample] * dt)
        accum = accum + T * (1 - alpha)
        T = T * alpha
    print(f"T.requires_grad: {T.requires_grad}")
    print(f"accum.requires_grad: {accum.requires_grad}")
    print(f"alpha.requires_grad: {alpha.requires_grad}")
    print(f"samplePoints.requires_grad: {samplePoints.requires_grad}")
    print(f"densities.requires_grad: {densities.requires_grad}")
    for name, param in neaf_model.named_parameters():
        print(f"{name} gradient: {param.grad}")

    return accum
