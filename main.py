import astra
import matplotlib.pyplot as plt
import numpy as np
from NeAF import *

# create geometries and projector
proj_geom = astra.create_proj_geom('fanflat', 1.0, 256, np.linspace(0, 2.0 * np.pi, 6, endpoint=False), 10000, 200)
vol_geom = astra.create_vol_geom(256, 256, -128, 128, -128, 128)
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

torch.set_grad_enabled(True)

neafModel = NeAF(numInputFeatures=2, encodingDegree=6)
neafModel.train()

for name, param in neafModel.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")

torchSino = torch.tensor(sinogram, dtype=torch.float32, requires_grad=True).flatten()
torchSino = torchSino.to(dtype=torch.float32)


loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(neafModel.parameters(), lr=0.1, momentum=0.9)

for epoch in range(10):
    optimizer.zero_grad()
    output = renderRays(neafModel, detectorPixels, detectorCount*projCount, 128)
    loss = loss_fn(output, torchSino)
    print(f"output.requires_grad: {output.requires_grad}")
    print(f"torchSino.requires_grad: {torchSino.requires_grad}")
    # for name, param in neafModel.named_parameters():
    #     if param.grad is None:
    #         print(f"No gradient for parameter: {name}")

    loss.backward()

    optimizer.step()
    print(f"Epoch {epoch}: loss {loss}")



output = output.cpu().detach().numpy()
output = np.reshape(output, [projCount, detectorCount])
print(np.max(output))
print(np.min(output))


# garbage disposal
astra.data2d.delete([sinogram_id, V_exact_id])
astra.projector.delete(proj_id)