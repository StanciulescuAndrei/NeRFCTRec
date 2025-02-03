import astra
import matplotlib.pyplot as plt
import numpy as np
from NeAF import *

def samplingGenerator(descriptor, numSamplePoints):
    samplePoints = []
    dt = []
    for px in descriptor['pixels']:
            sp = np.array(descriptor['src'])
            ep = np.array(px)

            recVolumeIntersections = getParametricIntersection(sp, ep, bboxMin, bboxMax)
            if recVolumeIntersections != None and len(recVolumeIntersections) == 2:
                sp, ep = recVolumeIntersections / (bboxMax - bboxMin) * 2.0
                dt.append(np.linalg.norm(ep - sp) / numSamplePoints)
            else:
                dt.append(0.0)
            
            for t in np.linspace(0, 1, numSamplePoints):
                samplePoints.append(sp * (1.0 - t) + ep * t)

    return samplePoints, dt

# create geometries and projector
bboxMin = np.array([-128, -128])
bboxMax = np.array([128, 128])

proj_geom = astra.create_proj_geom('fanflat', 1, 256, np.linspace(0, 2 * np.pi, 7, endpoint=False), 10000, 200)
vol_geom = astra.create_vol_geom(256, 256, bboxMin[0], bboxMax[0], bboxMin[1], bboxMax[1])
proj_id = astra.create_projector('cuda', proj_geom, vol_geom)

proj_geom_vec = astra.geom_2vec(proj_geom)

# generate phantom image
V_exact_id, V_exact = astra.data2d.shepp_logan(vol_geom)

# create forward projection
sinogram_id, sinogram = astra.create_sino(V_exact, proj_id)

detectorCount = proj_geom_vec['DetectorCount']
vectors = proj_geom_vec['Vectors']
projCount = len(vectors)


detectorPixels = []

for descriptor in  vectors:
    v = np.array(descriptor)
    rotationalArray = dict()
    rotationalArray['src'] = [v[0:2]]
    rotationalArray['pixels'] = []
    for i in range(0, detectorCount):
        rotationalArray['pixels'].append((i - detectorCount / 2.0) * v[4:6] + v[2:4])
    detectorPixels.append(rotationalArray)

numSamplePoints = 128

allSamplePoints = []

for descriptor in detectorPixels:
    allSamplePoints.append(tuple(samplingGenerator(descriptor, numSamplePoints)))

torch.set_grad_enabled(True)

neafModel = NeAF(numInputFeatures=2, encodingDegree=8).cuda()

neafModel.train()

torchSino = torch.tensor(sinogram, dtype=torch.float32, requires_grad=True).reshape([projCount * detectorCount]).cuda()

trainModel(neafModel, torchSino, allSamplePoints, detectorCount, projCount)

torch.save(neafModel.state_dict(), "checkpoint")

evalsamplePoints = np.zeros([256 * 256, 2], dtype=np.float32)
for x in range(256):
    for y in range(256):
        evalsamplePoints[x * 256 + y, 0] = x / 128.0 - 1.0
        evalsamplePoints[x * 256 + y, 1] = y / 128.0 - 1.0
evalSamples = torch.tensor(evalsamplePoints, requires_grad=False, dtype=torch.float32).cuda()

output, last_sino = sampleModel(neafModel, evalSamples, allSamplePoints, detectorCount, projCount)

plt.gray()
plt.subplot(1, 4, 1)
plt.imshow(V_exact)
plt.subplot(1, 4, 2)
plt.imshow(output)
plt.subplot(1, 4, 3)
plt.imshow(last_sino)
plt.subplot(1, 4, 4)
plt.imshow(sinogram)
plt.show()


# garbage disposal
astra.data2d.delete([sinogram_id, V_exact_id])
astra.projector.delete(proj_id)