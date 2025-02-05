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

    def getSamples(self, viewRange):
        t_param = torch.linspace(0, 1, self.numSamplePoints, device="cuda").view(1, self.numSamplePoints, 1)

        viewRangeStart = viewRange[0] * self.getDetectorCount()
        viewRangeEnd   = viewRange[1] * self.getDetectorCount()

        samplePoints = self.z_start[viewRangeStart:viewRangeEnd, None, :] + t_param * (self.z_end[viewRangeStart:viewRangeEnd, None, :] - self.z_start[viewRangeStart:viewRangeEnd, None, :])

        dt = torch.norm(self.z_end[viewRangeStart:viewRangeEnd, :] - self.z_start[viewRangeStart:viewRangeEnd, :], dim=1) / self.numSamplePoints

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

        self.z_start = torch.zeros([self.detectorCount * self.projCount, 2], device="cuda")
        self.z_end   = torch.zeros([self.detectorCount * self.projCount, 2], device="cuda")

        idx = 0
        for descriptor in self.detectorPixels:
            for px in descriptor['pixels']:
                sp = np.array(descriptor['src'])
                ep = np.array(px)

                recVolumeIntersections = self.getParametricIntersection(sp, ep, self.bboxMin, self.bboxMax)
                if recVolumeIntersections != None and len(recVolumeIntersections) == 2:
                    self.z_start[idx, :] = torch.tensor(recVolumeIntersections[0] / (self.bboxMax - self.bboxMin) * 2.0)
                    self.z_end[idx, :]   = torch.tensor(recVolumeIntersections[1] / (self.bboxMax - self.bboxMin) * 2.0)
                    idx += 1

    def getSinoNumberOfPixels(self):
        return self.projCount * self.detectorCount

    def getProjCount(self):
        return self.projCount
    
    def getDetectorCount(self):
        return self.detectorCount
    
    def getNumSamples(self):
        return self.numSamplePoints

            
def renderRays(neaf_model, scanningGeometry: ScanningGeometry, viewRange, shouldRanzomize):
    samplePoints, dt = scanningGeometry.getSamples(viewRange)
                
    samplePoints = samplePoints.flatten(0, 1)

    if shouldRanzomize:
        samplePoints += torch.rand(samplePoints.shape, device=torch.device('cuda')) * torch.min(dt) * 0.1
    densities = neaf_model(samplePoints)
    
    densities = densities.view( int(densities.shape[0] / scanningGeometry.getNumSamples()), scanningGeometry.getNumSamples())

    accum = torch.zeros(densities.shape[0], dtype=torch.float32, requires_grad=True).cuda()
    for sample in range(scanningGeometry.getNumSamples()):
        accum = accum + densities[:, sample] * dt 
    return accum

def trainModel(neafModel, groundTruth, scanningGeometry: ScanningGeometry):

    maxSamples = 256 * 6 * 128 # Experimental max samples fitting on the GPU

    viewsPerBatch = int(np.floor(maxSamples / (scanningGeometry.getDetectorCount() * scanningGeometry.getNumSamples())))

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(neafModel.parameters(), lr=0.001, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.8)

    neafModel.train(True)

    lossArray = []

    for epoch in range(100):
        runningLoss = 0
        viewRange = [0, viewsPerBatch]
        while viewRange[0] < scanningGeometry.getProjCount():
            optimizer.zero_grad()
            output = renderRays(neafModel, scanningGeometry, viewRange, True)

            loss = loss_fn(output, groundTruth[viewRange[0] * scanningGeometry.getDetectorCount():viewRange[1] * scanningGeometry.getDetectorCount()])
            loss.backward()
            runningLoss += loss.item()
            optimizer.step()

            viewRange = [viewRange[0] + viewsPerBatch, min(viewRange[1] + viewsPerBatch, scanningGeometry.getProjCount())]

        scheduler.step()
        lossArray.append(runningLoss / scanningGeometry.getProjCount())
        print(f"Epoch {epoch}: loss per projection {lossArray[-1]}")
    
    return lossArray

@torch.no_grad()
def sampleModel(neafModel, samples, scanningGeometry: ScanningGeometry):
    output = neafModel(samples).detach().cpu()

    maxSamples = 256 * 6 * 128 # Experimental max samples fitting on the GPU

    viewsPerBatch = int(np.floor(maxSamples / (scanningGeometry.getDetectorCount() * scanningGeometry.getNumSamples())))

    viewRange = [0, viewsPerBatch]
    sino = torch.zeros([256, scanningGeometry.getProjCount()])
    while viewRange[0] < scanningGeometry.getProjCount():
        partialSino = renderRays(neafModel, scanningGeometry, viewRange, False).detach().cpu()
        sino[:, viewRange[0]:viewRange[1]] = torch.reshape(partialSino, [viewRange[1] - viewRange[0], 256]).transpose(0, 1)
        viewRange = [viewRange[0] + viewsPerBatch, min(viewRange[1] + viewsPerBatch, scanningGeometry.getProjCount())]
    output = torchvision.transforms.functional.hflip(torch.reshape(output, [256, 256]))
    return torch.transpose(output, 0, 1), torch.transpose(sino, 0, 1)
    
