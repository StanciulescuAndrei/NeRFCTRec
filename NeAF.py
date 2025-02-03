import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
import math

class NeAF(nn.Module):
    def __init__(self, numInputFeatures, encodingDegree):
        super(NeAF, self).__init__()
        self.numInputFeatures = numInputFeatures
        self.encodingDegree = encodingDegree
        self.block1 = nn.Sequential(
            nn.Linear(2 * encodingDegree * self.numInputFeatures + 2, 256), nn.LeakyReLU(),
            nn.Linear(256, 256), nn.LeakyReLU(),
            # nn.Linear(256, 256), nn.LeakyReLU(),
            # nn.Linear(256, 256), nn.LeakyReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Linear(2 * encodingDegree * self.numInputFeatures + 2 + 256, 256), nn.LeakyReLU(),
            # nn.Linear(256, 256), nn.LeakyReLU(),
            # nn.Linear(256, 256), nn.LeakyReLU(),
            # nn.Linear(256, 256), nn.LeakyReLU(),
            nn.Linear(256, 128), nn.LeakyReLU(),
            nn.Linear(128, 32), nn.LeakyReLU(),
            nn.Linear(32, 1), nn.LeakyReLU()
        )

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
        return out
    

def getParametricIntersection(startPoint, endPoint, bboxMin, bboxMax):
    ray_dir = endPoint - startPoint
    ray_dir[ray_dir == 0] = 1e-8  # Avoid division by zero by adding a small epsilon

    t_min = (bboxMin - startPoint) / ray_dir
    t_max = (bboxMax - startPoint) / ray_dir

    # Swap t_min and t_max where necessary
    t1 = np.minimum(t_min, t_max)
    t2 = np.maximum(t_min, t_max)

    # Find the largest t_min and the smallest t_max
    t_near = t1.max()
    t_far = t2.min()

    # Check for intersection
    if t_near > t_far or t_far < 0:
        return None  # No intersection
    
    intersection_points = []
    if t_near >= 0:
        intersection_points.append(startPoint + t_near * ray_dir)
    if t_far >= 0:
        intersection_points.append(startPoint + t_far * ray_dir)

    return intersection_points

def renderRays(neaf_model, allSamplePoints, batchSize, numSamplePoints, shouldRanzomize):
    # numSamplePoints x pixelCount x projectionCount

    samplePoints = []
    dt = []
    for view in allSamplePoints:
        samplePoints += view[0]
        dt += view[1]
                
    samplePoints = torch.tensor(np.array(samplePoints), dtype=torch.float32, requires_grad=True).squeeze(1).cuda()
    dt = torch.tensor(np.array(dt), dtype=torch.float32, requires_grad=True).cuda()
    if shouldRanzomize:
        samplePoints += torch.rand(samplePoints.shape, device=torch.device('cuda')) * torch.min(dt) * 0.05
    densities = neaf_model(samplePoints)
    
    densities = densities.view(batchSize, numSamplePoints)

    accum = torch.zeros(batchSize, dtype=torch.float32, requires_grad=True).cuda()
    Tval = torch.ones(batchSize).cuda()
    for sample in range(numSamplePoints):
        accum = accum + densities[:, sample] * dt
        # alpha = torch.exp(-densities[:, sample] * dt).cuda()
        # accum = accum + Tval * (1 - alpha)
        # Tval = Tval * alpha
    return accum

def trainModel(neafModel, groundTruth, allSamplePoints, detectorCount, projCount):

    batchSize = 10
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(neafModel.parameters(), lr=0.001, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.8)

    neafModel.train(True)

    for epoch in range(1000):
        running_loss = 0
        for i in range(0, projCount, batchSize):
            restrictedBatch = min(batchSize, projCount - i)
            optimizer.zero_grad()
            output = renderRays(neafModel, allSamplePoints[i:i + restrictedBatch], detectorCount * restrictedBatch, 128, True)

            loss = loss_fn(output, groundTruth[detectorCount * i : detectorCount * (i + restrictedBatch)])
            loss.backward()

            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
        print(f"Epoch {epoch}: loss per projection {loss / projCount}")

@torch.no_grad()
def sampleModel(neafModel, samples, allSamplePoints, detectorCount, projCount):
    output = neafModel(samples).detach().cpu()
    sino = torch.zeros([256, projCount])
    batchSize = 5
    for i in range(0, projCount, batchSize):
        restrictedBatch = min(batchSize, projCount - i)
        sino[:, i:i + restrictedBatch] = torch.reshape(renderRays(neafModel, allSamplePoints[i:i + restrictedBatch], detectorCount * restrictedBatch, 128, False).detach().cpu(), [restrictedBatch, 256]).transpose(0, 1)
    output = torch.reshape(output, [256, 256])
    return torch.transpose(output, 0, 1), torch.transpose(sino, 0, 1)
    
