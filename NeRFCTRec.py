import astra
import matplotlib.pyplot as plt
import numpy as np
from utils import *
import time
from torcheval.metrics.functional import peak_signal_noise_ratio

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

    return vol_geom, proj_geom_vec, bboxMin, bboxMax, V_exact_id, V_exact, sinogram_id, sinogram, proj_id

def eval_sirt(vol_geom, proj_id, sinogram_id):
    recon_id = astra.data2d.create('-vol', vol_geom, 0)
    cfg = astra.astra_dict('SIRT_CUDA')
    cfg['ProjectorId'] = proj_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ReconstructionDataId'] = recon_id
    cfg['option'] = { 'MinConstraint': 0, 'MaxConstraint': 1 }
    cgls_id = astra.algorithm.create(cfg)
    astra.algorithm.run(cgls_id, 2000)
    V = astra.data2d.get(recon_id)

    # garbage disposal
    astra.data2d.delete(recon_id)
    astra.algorithm.delete(cgls_id)

    return V

def eval_cgls(vol_geom, proj_id, sinogram_id):
    recon_id = astra.data2d.create('-vol', vol_geom, 0)
    cfg = astra.astra_dict('CGLS_CUDA')
    cfg['ProjectorId'] = proj_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ReconstructionDataId'] = recon_id
    cgls_id = astra.algorithm.create(cfg)
    astra.algorithm.run(cgls_id, 2000)
    V = astra.data2d.get(recon_id)

    # garbage disposal
    astra.data2d.delete(recon_id)
    astra.algorithm.delete(cgls_id)

    return V

def startApp():
    resolution = 256
    numProjections = 36
    numPixels = 256
    numSamplePoints = 128

    vol_geom, proj_geom_vec, bboxMin, bboxMax, V_exact_id, V_exact, sinogram_id, sinogram, proj_id = setupGeometry(resolution, numProjections, numPixels)

    cgls_reconstruction =  eval_cgls(vol_geom, proj_id, sinogram_id)
    sirt_reconstruction =  eval_sirt(vol_geom, proj_id, sinogram_id)
    
    scanningGeometry = ScanningGeometry(proj_geom_vec, bboxMin, bboxMax, numSamplePoints)

    torch.set_grad_enabled(True)

    # neafModel = NeAF(numInputFeatures=2, encodingDegree=8).cuda()
    nhmodel = NHGrid(numInputFeatures=2, numGridFeatures=2, gridLevels=4, hashSize=2**16).cuda()

    # checkpoint = torch.load("checkpoint", map_location="cuda")
    # neafModel.load_state_dict(checkpoint)

    nhmodel.train()

    torchSino = torch.tensor(sinogram / 128.0, dtype=torch.float32, requires_grad=True).reshape([scanningGeometry.getSinoNumberOfPixels()]).cuda()

    startTime = time.time()
    lossArray = trainModel(nhmodel, torchSino, scanningGeometry)
    endTime = time.time()

    print(f"Training done in: {endTime - startTime} seconds...")
    torch.save(nhmodel.state_dict(), "checkpoint")

    output = sampleModel(nhmodel, resolution)

    nerf_quality = peak_signal_noise_ratio(output, torch.tensor(V_exact))
    cgsl_quality = peak_signal_noise_ratio(torch.tensor(cgls_reconstruction), torch.tensor(V_exact))
    sirt_quality = peak_signal_noise_ratio(torch.tensor(sirt_reconstruction), torch.tensor(V_exact))

    print(f"NeRF PSNR: {nerf_quality} dB")
    print(f"CGSL PSNR: {cgsl_quality} dB")
    print(f"SIRT PSNR: {sirt_quality} dB")


    plt.gray()
    plt.subplot(1, 4, 1)
    plt.title("Reference")
    plt.imshow(V_exact)
    plt.subplot(1, 4, 2)
    plt.title("NeRF")
    plt.imshow(output)
    plt.subplot(1, 4, 3)
    plt.title("CGSL")
    plt.imshow(cgls_reconstruction)
    plt.subplot(1, 4, 4)
    plt.title("SIRT")
    plt.imshow(sirt_reconstruction)
    plt.suptitle("Reconstruction comparison", fontsize=14)
    # plt.subplot(1, 4, 3)
    # plt.imshow(last_sino)
    # plt.subplot(1, 4, 4)
    # plt.imshow(sinogram)
    plt.show()


    # garbage disposal
    astra.data2d.delete([sinogram_id, V_exact_id])
    astra.projector.delete(proj_id)

if __name__=="__main__":
    startApp()