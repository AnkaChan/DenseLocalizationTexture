import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import itertools
from os.path import join
from SkelFit.Visualization import obj2vtkFolder
from pathlib import Path
import tqdm

mtlHeader = \
'''newmtl material_0
Ka 0.200000 0.200000 0.200000
Kd 1.000000 1.000000 1.000000
Ks 1.000000 1.000000 1.000000
Tr 1.000000
illum 2
Ns 0.000000
map_Kd '''

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

def writeOBj(outObj, N, X, Y, Z, withMtl=True, textureFile=None):
    file  = open(outObj, 'w')
    fp = Path(outObj)
    # i is x, j is y
    # u = i / X.shape[0]
    # v = (X.shape[1]-j) / X.shape[1]
    if withMtl:
        outMtlFile = join(str(fp.parent), fp.stem + '.mtl')
        file.write('mtllib ./' + fp.stem + '.mtl\n')
        with open(outMtlFile, 'w') as fMtl:
            mtlStr = mtlHeader
            mtlStr += textureFile
            fMtl.write(mtlStr)
    for i, j in itertools.product(range(N), range(N)):
        u = i / X.shape[0]
        v = (X.shape[1]-j) / X.shape[1]
        file.write('v %f %f %f\n' %( X[i, j],  Y[i,j], Z[i,j] ))
        file.write('vt %f %f\n' %( u, v ))
    if withMtl:
        file.write('usemtl material_0\n')

    for i, j in itertools.product(range(0, N-1), range(1, N)):
        vId = j + i *N
        file.write('f %d/%d %d/%d %d/%d\n' %(vId, vId, vId+1, vId+1, vId+N+1, vId+N+1))
        file.write('f %d/%d %d/%d %d/%d\n' %(vId, vId, vId+N+1, vId+N+1, vId+N, vId+N))



if __name__ == '__main__':
    # outFile = r'F:\WorkingCopy2\2021_RandomlyDeformedGridMesh\Deformed\GridMesh_200x200.obj'
    outFolder = r'F:\WorkingCopy2\2021_RandomlyDeformedGridMesh\Deformed'
    textureFile = r"C:\Code\MyRepo\03_capture\DenseLocalizationTexture\S02_TextureDesign\RandGNoise_NN_2000x2000_WithMarkers.bmp"

    numMeshes = 1000
    sigmaBase = 10
    sigmaMultiplier = 10
    maxNumOfGuassians = 50
    gaussianMultiplier = 0.1
    N = 200
    X = np.linspace(-1, 1, N)
    Y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(X, Y)

    for iM in tqdm.tqdm(range(numMeshes)):
        # Mean vector and covariance matrix
        Z = np.zeros((200, 200))

        numG = np.random.randint(1, maxNumOfGuassians)
        # Pack X and Y into a single 3-dimensional array
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y
        for iG in range(numG):
            mu = np.array(np.random.rand(2))*2-1
            Sigma = np.array([[1., 0.5* (2*np.random.rand(1)-1)[0]], [0.5* (2*np.random.rand(1)-1)[0], 1.]]) / (sigmaBase+sigmaMultiplier*np.random.rand(1)-1)[0]
            Z1 = multivariate_gaussian(pos, mu, Sigma)

            Z = Z + Z1 * gaussianMultiplier


        outFile = join(outFolder, 'M' + str(iM).zfill(5) + '.obj')
        writeOBj(outFile, N, X, Y, Z, textureFile=textureFile)

    obj2vtkFolder(outFolder, )