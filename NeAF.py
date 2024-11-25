import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
import wandb

class NeAF(nn.Module):
    def __init__(self, numInputFeatures, encodingDegree):
        super(NeAF, self).__init__()
        self.numInputFeatures = numInputFeatures
        self.encodingDegree = encodingDegree
        self.block1 = nn.Sequential(
            nn.Linear(2 * encodingDegree * self.numInputFeatures + 2, 256), nn.LeakyReLU(),
            nn.Linear(256, 256), nn.LeakyReLU(),
            nn.Linear(256, 256), nn.LeakyReLU(),
            nn.Linear(256, 256), nn.LeakyReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Linear(2 * encodingDegree * self.numInputFeatures + 2 + 256, 256), nn.LeakyReLU(),
            nn.Linear(256, 256), nn.LeakyReLU(),
            nn.Linear(256, 256), nn.LeakyReLU(),
            nn.Linear(256, 256), nn.LeakyReLU(),
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

def renderRays(neaf_model, geometryDescriptor, batchSize, numSamplePoints, bboxMin, bboxMax):
    # numSamplePoints x pixelCount x projectionCount

    samplePoints = []
    dt = []

    rng = np.random.default_rng()

    def samplingGenerator(descriptor, rng, samplePoints, dt):
        for px in descriptor['pixels']:
                sp = np.array(descriptor['src'])
                ep = np.array(px)

                recVolumeIntersections = getParametricIntersection(sp, ep, bboxMin, bboxMax)
                disturbAmount = 0.05
                if recVolumeIntersections != None and len(recVolumeIntersections) == 2:
                    sp, ep = recVolumeIntersections / (bboxMax - bboxMin) * 2.0
                    disturbAmount = disturbAmount * np.linalg.norm(ep - sp) / numSamplePoints
                    dt.append(np.linalg.norm(ep - sp) / numSamplePoints)
                else:
                    dt.append(0.0)
                
                for t in np.linspace(0, 1, numSamplePoints):
                    samplePoints.append(sp * (1.0 - t) + ep * t + disturbAmount * dt[-1] * rng.standard_normal(2))

        return samplePoints, dt

    if type(geometryDescriptor) == type(list()):
        for desc in geometryDescriptor:
            samplePoints, dt = samplingGenerator(desc, rng, samplePoints, dt)
    else:
        samplePoints, dt = samplingGenerator(geometryDescriptor, rng, samplePoints, dt)
                

    samplePoints = torch.tensor(np.array(samplePoints), dtype=torch.float32, requires_grad=True).squeeze(1).cuda()
    densities = neaf_model(samplePoints)
    dt = torch.tensor(np.array(dt), dtype=torch.float32, requires_grad=True).cuda()

    densities = densities.view(batchSize, numSamplePoints)

    accum = torch.zeros(batchSize, dtype=torch.float32, requires_grad=True).cuda()
    Tval = torch.ones(batchSize).cuda()
    for sample in range(numSamplePoints):
        accum = accum + densities[:, sample] * dt
        # alpha = torch.exp(-densities[:, sample] * dt).cuda()
        # accum = accum + Tval * (1 - alpha)
        # Tval = Tval * alpha
    return accum

def trainModel(neafModel, groundTruth, detectorPixels, detectorCount, projCount, bboxMin, bboxMax):

    batchSize = 5
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(neafModel.parameters(), lr=0.001, momentum=0.9)

    neafModel.train(True)

    for epoch in range(500):
        running_loss = 0
        for i in range(0, projCount, batchSize):
            restrictedBatch = min(batchSize, projCount - i)
            optimizer.zero_grad()
            output = renderRays(neafModel, detectorPixels[i:i + restrictedBatch], detectorCount * restrictedBatch, 128, bboxMin, bboxMax)

            loss = loss_fn(output, groundTruth[detectorCount * i : detectorCount * (i + restrictedBatch)])
            loss.backward()

            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch}: loss {loss}")

@torch.no_grad()
def sampleModel(neafModel, samples, detectorPixels, detectorCount, projCount, bboxMin, bboxMax):
    output = neafModel(samples).detach().cpu()
    sino = torch.zeros([256, projCount])
    for i in range(projCount):
        sino[:, i] = renderRays(neafModel, detectorPixels[i], detectorCount, 128, bboxMin, bboxMax).detach().cpu()
    output = torch.reshape(output, [256, 256])
    return torch.transpose(output, 0, 1), torch.transpose(sino, 0, 1)
    
