import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import math
import astra
import torchvision.transforms.functional

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
    
class ScanningGeometry:
    @staticmethod
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

    def samplingGenerator(self, descriptor, numSamplePoints):
        samplePoints = []
        dt = []
        for px in descriptor['pixels']:
                sp = np.array(descriptor['src'])
                ep = np.array(px)

                recVolumeIntersections = self.getParametricIntersection(sp, ep, self.bboxMin, self.bboxMax)
                if recVolumeIntersections != None and len(recVolumeIntersections) == 2:
                    sp, ep = recVolumeIntersections / (self.bboxMax - self.bboxMin) # * 2.0
                    dt.append(np.linalg.norm(ep - sp) / numSamplePoints)
                else:
                    dt.append(0.0)
                
                for t in np.linspace(0, 1, numSamplePoints):
                    samplePoints.append(sp * (1.0 - t) + ep * t)

        return samplePoints, dt

    def __init__(self, projectorGeometryVector, bboxMin, bboxMax, numSamplePoints):
        self.bboxMin = bboxMin
        self.bboxMax = bboxMax
        self.numSamplePoints = numSamplePoints

        self.detectorCount = projectorGeometryVector['DetectorCount']
        self.vectors = projectorGeometryVector['Vectors']
        self.projCount = len(self.vectors)

        self.detectorPixels = []

        for descriptor in  self.vectors:
            v = np.array(descriptor)
            rotationalArray = dict()
            rotationalArray['src'] = [v[0:2]]
            rotationalArray['pixels'] = []
            for i in range(0, self.detectorCount):
                rotationalArray['pixels'].append((i - self.detectorCount / 2.0) * v[4:6] + v[2:4])
            self.detectorPixels.append(rotationalArray)

        self.allSamplePoints = []

        for descriptor in self.detectorPixels:
            self.allSamplePoints.append(tuple(self.samplingGenerator(descriptor, self.numSamplePoints)))

    def getSinoNumberOfPixels(self):
        return self.projCount * self.detectorCount

    def getProjCount(self):
        return self.projCount
    
    def getDetectorCount(self):
        return self.detectorCount
    
    def getSamplePoints(self):
        return self.allSamplePoints
    
    def getNumSamplePoints(self):
        return self.numSamplePoints
            
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
        samplePoints += torch.rand(samplePoints.shape, device=torch.device('cuda')) * torch.min(dt) * 0.1
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

def trainModel(neafModel, groundTruth, scanningGeometry: ScanningGeometry):

    batchSize = 8
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(neafModel.parameters(), lr=0.001, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.8)

    neafModel.train(True)

    lossArray = []

    for epoch in range(600):
        running_loss = 0
        for i in range(0, scanningGeometry.getProjCount(), batchSize):
            restrictedBatch = min(batchSize, scanningGeometry.getProjCount() - i)
            optimizer.zero_grad()
            output = renderRays(neafModel, scanningGeometry.getSamplePoints()[i:i + restrictedBatch], scanningGeometry.getDetectorCount() * restrictedBatch, scanningGeometry.getNumSamplePoints(), True)

            loss = loss_fn(output, groundTruth[scanningGeometry.getDetectorCount() * i : scanningGeometry.getDetectorCount() * (i + restrictedBatch)])
            loss.backward()

            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
        lossArray.append(running_loss / scanningGeometry.getProjCount())
        print(f"Epoch {epoch}: loss per projection {lossArray[-1]}")
    
    return lossArray

@torch.no_grad()
def sampleModel(neafModel, samples, scanningGeometry: ScanningGeometry):
    output = neafModel(samples).detach().cpu()
    sino = torch.zeros([256, scanningGeometry.getProjCount()])
    batchSize = 2
    for i in range(0, scanningGeometry.getProjCount(), batchSize):
        restrictedBatch = min(batchSize, scanningGeometry.getProjCount() - i)
        sino[:, i:i + restrictedBatch] = torch.reshape(renderRays(neafModel, scanningGeometry.getSamplePoints()[i:i + restrictedBatch], scanningGeometry.getDetectorCount() * restrictedBatch, scanningGeometry.getNumSamplePoints(), False).detach().cpu(), [restrictedBatch, 256]).transpose(0, 1)
    output = torchvision.transforms.functional.hflip(torch.reshape(output, [256, 256]))
    return torch.transpose(output, 0, 1), torch.transpose(sino, 0, 1)
    
