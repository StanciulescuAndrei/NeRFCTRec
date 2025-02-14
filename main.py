import astra
import matplotlib.pyplot as plt
import numpy as np
from utils import *


def setupGeometry(resolution, numProjections, numPixels):
    # create geometries and projector
    bboxMin = np.array([-resolution // 2, -resolution // 2])
    bboxMax = np.array([resolution // 2, resolution // 2])

    proj_geom = astra.create_proj_geom('fanflat', 1, numPixels, np.linspace(0, 2 * np.pi, numProjections, endpoint=False), 10000, 200)
    vol_geom = astra.create_vol_geom(resolution, resolution, bboxMin[0], bboxMax[0], bboxMin[1], bboxMax[1])
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)

    proj_geom_vec = astra.geom_2vec(proj_geom)

    # generate phantom image
    V_exact_id, V_exact = astra.data2d.shepp_logan(vol_geom)

    # create forward projection
    sinogram_id, sinogram = astra.create_sino(V_exact, proj_id)

    return proj_geom_vec, bboxMin, bboxMax, V_exact_id, V_exact, sinogram_id, sinogram, proj_id

def main():
    resolution = 256
    numProjections = 36
    numPixels = 256
    numSamplePoints = 64

    proj_geom_vec, bboxMin, bboxMax, V_exact_id, V_exact, sinogram_id, sinogram, proj_id = setupGeometry(resolution, numProjections, numPixels)
    
    scanningGeometry = ScanningGeometry(proj_geom_vec, bboxMin, bboxMax, numSamplePoints)

    torch.set_grad_enabled(True)
    neafModel = NeAF(numInputFeatures=2, encodingDegree=8).cuda()

    nhmodel = NHGrid(numInputFeatures=2, numGridFeatures=2, gridLevels=4, hashSize=2**12).cuda()

    # checkpoint = torch.load("checkpoint", map_location="cuda")
    # neafModel.load_state_dict(checkpoint)

    nhmodel.train()

    torchSino = torch.tensor(sinogram / 128.0, dtype=torch.float32, requires_grad=True).reshape([scanningGeometry.getSinoNumberOfPixels()]).cuda()

    lossArray = trainModel(nhmodel, torchSino, scanningGeometry)

    torch.save(nhmodel.state_dict(), "checkpoint")


    plt.plot(lossArray)
    plt.show()

    output = sampleModel(nhmodel, resolution)

    plt.gray()
    plt.subplot(1, 2, 1)
    plt.title("Reference")
    plt.imshow(V_exact)
    plt.subplot(1, 2, 2)
    plt.title("Reconstruction")
    plt.imshow(output)
    plt.suptitle("Classic NeRF Reconstruction", fontsize=14)
    # plt.subplot(1, 4, 3)
    # plt.imshow(last_sino)
    # plt.subplot(1, 4, 4)
    # plt.imshow(sinogram)
    plt.show()


    # garbage disposal
    astra.data2d.delete([sinogram_id, V_exact_id])
    astra.projector.delete(proj_id)

if __name__=="__main__":
    main()