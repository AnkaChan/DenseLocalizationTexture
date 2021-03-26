import cv2, glob, itertools, json
import numpy as np
from os.path import join
import os, tqdm
from pathlib import Path
import OpenEXR
import Imath, array

from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import imgaug.augmenters as iaa

def genAugImgCrops(img, keypoints, augSeq, cropSize, padSize):
    hcropSize = int(cropSize / 2)

    image_padded = cv2.copyMakeBorder(img, padSize, padSize, padSize, padSize, cv2.BORDER_CONSTANT)

    kps = [
        Keypoint(x=c['coord'][0] + padSize, y=c['coord'][1] + padSize) for c in keypoints
        # left eye (from camera perspective)
    ]

    imgSize = image_padded.shape

    kpsoi = KeypointsOnImage(kps, shape=imgSize)

    image_aug, kpsoi_aug = augSeq(image=image_padded, keypoints=kpsoi)

    kpArr = kpsoi_aug.to_xy_array()

    imgW = image_aug.shape[1]
    imgH = image_aug.shape[0]

    imgCrops = []
    corners = []
    for iP in range(kpArr.shape[0]):
        pProj = kpArr[iP, :]
        if pProj[0] > hcropSize and pProj[0] < imgW - 1 - hcropSize and pProj[1] > hcropSize and pProj[
            1] < imgH - 1 - hcropSize:
            c = [pProj[0], pProj[1]]
            cxi = int(c[0]) if c[0] - np.floor(c[0]) <= 0.5 else int(c[0]) + 1
            cyi = int(c[1]) if c[1] - np.floor(c[1]) <= 0.5 else int(c[1]) + 1

            cornerCrop = image_aug[cyi - hcropSize:cyi + hcropSize + 1,
                         cxi - hcropSize:cxi + hcropSize + 1, np.newaxis]

            imgCrops.append(cornerCrop)
            corners.append(keypoints[iP])

    return imgCrops, corners

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
    file.close()
    return np.stack([u, v], axis=-1)

def getMarkersToVertsCorrs(markerStride, numGrid, ):
    '''
    :param markerStride: distance between markers in vertical or horizontal direction, the unit is number of grids
    :param numGrid: number of total grids in vertical or horizontal direction
    :return: the mesh vertex id each marker corresponds to
    '''
    markerToVIdCorrs=[]
    for i, j in itertools.product(range(markerStride, numGrid, markerStride), range(markerStride, numGrid, markerStride)):
        vertId = i*numGrid + j
        markerToVIdCorrs.append(vertId)

    return markerToVIdCorrs


if __name__ == '__main__':
    # Figure out the markers' corresponded vertex id
    inputVertCoordFolder = r'F:\WorkingCopy2\2021_03_01_RandomlyDeformedGridMesh\Rendered\VertexPositions'
    inputImgFolder = r'F:\WorkingCopy2\2021_03_01_RandomlyDeformedGridMesh\Rendered\Scene'
    inputUVFolder = r'F:\WorkingCopy2\2021_03_01_RandomlyDeformedGridMesh\Rendered\UV'

    # outFolder = r'F:\WorkingCopy2\2021_RandomlyDeformedGridMesh\Data'
    outFolder = r'F:\WorkingCopy2\2021_03_01_RandomlyDeformedGridMesh\Data128_withRotation'

    inputVertCoordFiles = sorted(glob.glob(join(inputVertCoordFolder, '*.json')))
    inputImgFiles =  sorted(glob.glob(join(inputImgFolder, '*.jpg')))
    inputUVFiles = sorted(glob.glob(join(inputUVFolder, '*.exr')))
    # inputUVFiles = sorted(glob.glob(join(inputUVFolder, '*.jpg')))

    os.makedirs(outFolder, exist_ok=True)

    numGrid = 200
    numMarkers = 20
    markerStride = 10
    cropSize = 128
    halfCropSize = int(cropSize/2)

    markersToPreserve = [199]

    augCfg = {
        # 'mul': (0.8, 1.2),
        'r': (-180, 180),
        # 'shear': (-30, 30),
        # 'useStar': 0,
        'numAugs': 10,
        # 'cropSize': 75,
    }
    seq_img = iaa.Sequential([
        # iaa.ElasticTransformation(alpha=500, sigma=50),
        # iaa.Multiply(augCfg['mul']),
        iaa.Affine(rotate=augCfg['r'],  order=[0])
        # iaa.AddToHueAndSaturation((-10, 10))  # color jitter, only affects the image
    ])

    seq_uv = iaa.Sequential([
        # iaa.ElasticTransformation(alpha=500, sigma=50),
        # iaa.Multiply(augCfg['mul']),
        iaa.Affine(rotate=augCfg['r'],  order=[1])
        # iaa.AddToHueAndSaturation((-10, 10))  # color jitter, only affects the image
    ])


    markerToVIdCorrs = getMarkersToVertsCorrs(markerStride, numGrid)

    dataImgPerMarkers = [[] for iM in range(len(markersToPreserve))]
    dataUVPerMarkers = [[] for iM in range(len(markersToPreserve))]

    iImg = 0
    for inputVCoordFile, imgFile, uvFile in tqdm.tqdm(zip(inputVertCoordFiles, inputImgFiles, inputUVFiles)):
        assert Path(imgFile).stem == Path(uvFile).stem

        vCoords = json.load(open(inputVCoordFile))
        vCoords = [vCoords[i] for i in markerToVIdCorrs]
        img = cv2.imread(imgFile)
        uv = readUV(uvFile)
        # uv = cv2.imread(uvFile)

        # vCoord = [round(vCoord[0]), imgSize[0] - round(vCoord[1])]
        imgSize = img.shape

        kps = [
            Keypoint(x=c[0] , y=imgSize[0] - c[1] ) for c in vCoords
            # left eye (from camera perspective)
        ]

        imgSize = img.shape
        kpsoi = KeypointsOnImage(kps, shape=imgSize)

        images = img[None, ...]
        uv = uv[None, ...]
        for iAug in range(augCfg['numAugs']):
            seq_img = seq_img.localize_random_state()

            seq_img_i = seq_img.to_deterministic()
            seq_uv_i = seq_uv.to_deterministic()

            seq_uv_i = seq_uv_i.copy_random_state(seq_img_i, matching="name")

            # imgs_aug = seq_img_i.augment_images(images)
            imgs_aug, kpsoi_aug = seq_img_i(images=images, keypoints=kpsoi)
            vCoordsAug = kpsoi_aug.to_xy_array()

            uvs_aug = seq_uv_i.augment_images(uv)

            # cv2.imwrite(join(outFolder, Path(imgFile).stem+"_img" + str(iAug) + ".bmp"), np.squeeze(imgs_aug))
            # cv2.imwrite(join(outFolder, Path(imgFile).stem+"_uv" + str(iAug) + ".bmp"), np.squeeze(uvs_aug))

            for markerIndex, markerId in enumerate(markersToPreserve):
                vCoord = [int(vCoordsAug[markerId][0]), int(vCoordsAug[markerId][1])]
                if vCoord[1]-halfCropSize > 0 and vCoord[1]+halfCropSize < img.shape[0] and  vCoord[0]-halfCropSize > 0 and  vCoord[0]+halfCropSize < img.shape[1]:
                    crop = np.array(imgs_aug[0, vCoord[1]-halfCropSize:vCoord[1]+halfCropSize, vCoord[0]-halfCropSize: vCoord[0]+halfCropSize, :])
                    cropUV = np.array(uvs_aug[0, vCoord[1]-halfCropSize:vCoord[1]+halfCropSize, vCoord[0]-halfCropSize: vCoord[0]+halfCropSize, :])
                    # cv2.imshow('crop', crop)
                    # cv2.imshow('u', cropUV[:,:,0])
                    # cv2.imshow('v', cropUV[:,:,1])
                    # cv2.waitKey()

                    dataImgPerMarkers[markerIndex].append(crop)
                    dataUVPerMarkers[markerIndex].append(cropUV)

            if not iImg%100 and iImg:
                for markerIndex, markerId in enumerate(markersToPreserve):
                    outImgDataFile = join(outFolder, 'ImgMarker_' + str(markerId).zfill(3) + '.npy')
                    outUVDataFile = join(outFolder, 'UVMarker_' + str(markerId).zfill(3) + '.npy')
                    np.save(outImgDataFile, np.stack(dataImgPerMarkers[markerIndex]))
                    np.save(outUVDataFile, np.stack(dataUVPerMarkers[markerIndex]))
            iImg+=1

    for markerIndex, markerId in enumerate(markersToPreserve):
        outImgDataFile = join(outFolder, 'ImgMarker_' + str(markerId).zfill(3) + '.npy')
        outUVDataFile = join(outFolder, 'UVMarker_' + str(markerId).zfill(3) + '.npy')
        np.save(outImgDataFile, np.stack(dataImgPerMarkers[markerIndex]))
        np.save(outUVDataFile, np.stack(dataUVPerMarkers[markerIndex]))
