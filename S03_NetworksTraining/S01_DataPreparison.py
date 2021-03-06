import cv2, glob, itertools, json
import numpy as np
from os.path import join
import os, tqdm
from pathlib import Path
import OpenEXR
import Imath, array

def readUV(inFile):
    file = OpenEXR.InputFile(inFile)

    # Compute the size
    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R, G, B, Z) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B", "Z")]
    # (R, G, B, Z) = [array.array('f', file.channel(Chan, FLOAT)) for Chan in ("R", "G", "B", "Z")]

    u = np.array(R)
    v = np.array(G)

    u.resize(sz[1], sz[0])
    v.resize(sz[1], sz[0])
    # cv2.imshow('u', u)
    # cv2.waitKey()
    return np.stack([u, v], axis=-1)

if __name__ == '__main__':
    # Figure out the markers' corresponded vertex id
    inputVertCoordFolder = r'F:\WorkingCopy2\2021_03_01_RandomlyDeformedGridMesh\Rendered\VertexPositions'
    inputImgFolder = r'F:\WorkingCopy2\2021_03_01_RandomlyDeformedGridMesh\Rendered\Scene'
    inputUVFolder = r'F:\WorkingCopy2\2021_03_01_RandomlyDeformedGridMesh\Rendered\UV'

    # outFolder = r'F:\WorkingCopy2\2021_RandomlyDeformedGridMesh\Data'
    outFolder = r'F:\WorkingCopy2\2021_03_01_RandomlyDeformedGridMesh\Data128'

    inputVertCoordFiles = sorted(glob.glob(join(inputVertCoordFolder, '*.json')))
    inputImgFiles =  sorted(glob.glob(join(inputImgFolder, '*.jpg')))
    inputUVFiles = sorted(glob.glob(join(inputUVFolder, '*.exr')))

    os.makedirs(outFolder, exist_ok=True)

    numGrid = 200
    numMarkers = 20
    markerStride = 10
    cropSize = 128
    halfCropSize = int(cropSize/2)

    numMarkersToPreserve = 50

    # vertexCoords[vIdToMakersCorrs, :] = makerCoords
    vIdToMakersCorrs = []
    for i, j in itertools.product(range(markerStride, numGrid, markerStride), range(markerStride, numGrid, markerStride)):
        vertId = i*numGrid + j
        vIdToMakersCorrs.append(vertId)

    vIdToMakersCorrs = vIdToMakersCorrs[:numMarkersToPreserve]

    dataImgPerMarkers = [[] for iM in range(len(vIdToMakersCorrs))]
    dataUVPerMarkers = [[] for iM in range(len(vIdToMakersCorrs))]

    for inputVCoordFile, imgFile, uvFile in tqdm.tqdm(zip(inputVertCoordFiles, inputImgFiles, inputUVFiles)):
        assert Path(imgFile).stem == Path(uvFile).stem

        vCoords = json.load(open(inputVCoordFile))
        vCoords = [vCoords[i] for i in vIdToMakersCorrs]
        img = cv2.imread(imgFile)
        uv = readUV(uvFile)
        # cv2.imshow('u', uv[:,:,0])
        # cv2.imshow('v', uv[:,:,1])
        # cv2.waitKey()
        for iMarker, vCoord in enumerate(vCoords):
            imgSize = img.shape

            vCoord = [round(vCoord[0]), imgSize[0] - round(vCoord[1])]
            # print(vCoord)

            if vCoord[1]-halfCropSize > 0 and vCoord[1]+halfCropSize < img.shape[0] and  vCoord[0]-halfCropSize > 0 and  vCoord[0]+halfCropSize < img.shape[1]:
                crop = img[vCoord[1]-halfCropSize:vCoord[1]+halfCropSize, vCoord[0]-halfCropSize: vCoord[0]+halfCropSize, :]
                cropUV = uv[vCoord[1]-halfCropSize:vCoord[1]+halfCropSize, vCoord[0]-halfCropSize: vCoord[0]+halfCropSize, :]
                # cv2.imshow('crop', crop)
                # cv2.imshow('u', cropUV[:,:,0])
                # cv2.imshow('v', cropUV[:,:,1])
                # cv2.waitKey()

                dataImgPerMarkers[iMarker].append(crop)
                dataUVPerMarkers[iMarker].append(cropUV)

    for iM in range(len(vIdToMakersCorrs)):
        outImgDataFile = join(outFolder, 'ImgMarker_' + str(iM).zfill(3) + '.npy')
        outUVDataFile = join(outFolder, 'UVMarker_' + str(iM).zfill(3) + '.npy')
        np.save(outImgDataFile, np.stack(dataImgPerMarkers[iM]))
        np.save(outUVDataFile, np.stack(dataUVPerMarkers[iM]))
