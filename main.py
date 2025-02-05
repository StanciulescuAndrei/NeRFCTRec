import astra
import matplotlib.pyplot as plt
import numpy as np
from NeAF import *

# create geometries and projector
bboxMin = np.array([-128, -128])
bboxMax = np.array([128, 128])

proj_geom = astra.create_proj_geom('fanflat', 1, 256, np.linspace(0, 2 * np.pi, 36, endpoint=False), 10000, 200)
vol_geom = astra.create_vol_geom(256, 256, bboxMin[0], bboxMax[0], bboxMin[1], bboxMax[1])
proj_id = astra.create_projector('cuda', proj_geom, vol_geom)

proj_geom_vec = astra.geom_2vec(proj_geom)

# generate phantom image
V_exact_id, V_exact = astra.data2d.shepp_logan(vol_geom)

# create forward projection
sinogram_id, sinogram = astra.create_sino(V_exact, proj_id)

numSamplePoints = 128

scanningGeometry = ScanningGeometry(proj_geom_vec, bboxMin, bboxMax, numSamplePoints)

torch.set_grad_enabled(True)

neafModel = NeAF(numInputFeatures=2, encodingDegree=8).cuda()

neafModel.train()

torchSino = torch.tensor(sinogram, dtype=torch.float32, requires_grad=True).reshape([scanningGeometry.getSinoNumberOfPixels()]).cuda()

lossArray = trainModel(neafModel, torchSino, scanningGeometry)

torch.save(neafModel.state_dict(), "checkpoint")


plt.plot(lossArray)
plt.show()

evalsamplePoints = np.zeros([256 * 256, 2], dtype=np.float32)
for x in range(256):
    for y in range(256):
        evalsamplePoints[x * 256 + y, 0] = x / 128.0 - 1.0
        evalsamplePoints[x * 256 + y, 1] = y / 128.0 - 1.0
evalSamples = torch.tensor(evalsamplePoints, requires_grad=False, dtype=torch.float32).cuda()

output, last_sino = sampleModel(neafModel, evalSamples, scanningGeometry)

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