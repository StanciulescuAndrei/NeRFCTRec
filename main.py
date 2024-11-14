import astra
import matplotlib.pyplot as plt
import numpy

# create geometries and projector
proj_geom = astra.create_proj_geom('fanflat', 1.0, 256, numpy.linspace(0, numpy.pi, 90, endpoint=False), 10000, 200)
vol_geom = astra.create_vol_geom(256,256)
proj_id = astra.create_projector('cuda', proj_geom, vol_geom)

# generate phantom image
V_exact_id, V_exact = astra.data2d.shepp_logan(vol_geom)

# create forward projection
sinogram_id, sinogram = astra.create_sino(V_exact, proj_id)

# reconstruct
recon_id = astra.data2d.create('-vol', vol_geom, 0)
cfg = astra.astra_dict('CGLS_CUDA')
cfg['ProjectorId'] = proj_id
cfg['ProjectionDataId'] = sinogram_id
cfg['ReconstructionDataId'] = recon_id
cgls_id = astra.algorithm.create(cfg)
astra.algorithm.run(cgls_id, 100)
V = astra.data2d.get(recon_id)
plt.gray()
plt.imshow(V)
plt.show()

# garbage disposal
astra.data2d.delete([sinogram_id, recon_id, V_exact_id])
astra.projector.delete(proj_id)
astra.algorithm.delete(cgls_id)