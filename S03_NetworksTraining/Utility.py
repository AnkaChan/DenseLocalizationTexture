import numpy as np
import cv2, json
from os.path import join
import random


def predToRGB(outputs, normalize=False):
    rgb = np.stack([outputs[..., 0], outputs[..., 1], np.zeros(outputs[..., 1].shape)], -1)

    if normalize:
        for iChannel in range(2):
            minVal = np.min(rgb[:,:,iChannel])
            maxVal = np.max(rgb[:,:,iChannel])
            rgb[:,:,iChannel] = (rgb[:,:,iChannel]-minVal) / (maxVal - minVal)

    return rgb

def predAndGdToRGB(outputs, gd, normalize=False):
    rgb = np.stack([outputs[..., 0], outputs[..., 1], np.zeros(outputs[..., 1].shape)], -1)
    gd = np.stack([gd[..., 0], gd[..., 1], np.zeros(gd[..., 1].shape)], -1)

    if normalize:
        for iChannel in range(2):
            minVal = np.min([np.min(rgb[:,:,iChannel]), np.min(gd[:,:,iChannel])])

            maxVal = np.max([np.max(rgb[:,:,iChannel]), np.max(gd[:,:,iChannel])])
            rgb[:,:,iChannel] = (rgb[:,:,iChannel]-minVal) / (maxVal - minVal)

            gd[:,:,iChannel] = (gd[:,:,iChannel]-minVal) / (maxVal - minVal)

    return rgb, gd

def loadDataV1(dataFolder, numMarkers, ):
    imgsPerMaker = []
    uvsPerMaker = []

    for iM in range(numMarkers):
        outImgDataFile = join(dataFolder, 'ImgMarker_' + str(iM).zfill(3) + '.npy')
        outUVDataFile = join(dataFolder, 'UVMarker_' + str(iM).zfill(3) + '.npy')

        img = np.load(outImgDataFile)
        uv = np.load(outUVDataFile)

        imgsPerMaker.append(img)
        uvsPerMaker.append(uv)

    return imgsPerMaker, uvsPerMaker

def shuffleTogether(a, b):
    # c = list(zip(a, b))
    # random.shuffle(c)
    # a, b = zip(*c)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def divideTrainValSet(imgs, uvs, numTest, shuffle=False):
    numData = imgs.shape[0]
    sizeTrain = numData - numTest

    if shuffle:
        imgs, uvs = shuffleTogether(imgs, uvs)

    trainData = imgs[:-numTest, ...]
    trainUV = uvs[:-numTest, ...]

    testData = imgs[-numTest:, ...]
    testUV = uvs[-numTest:, ...]

    return trainData, trainUV, testData, testUV, numData

def divideTrainValSetAll(imgs, uvs, numTest, shuffle=False):
    numDataAll = 0
    trainDataAll = []
    trainUVAll = []
    testDataAll = []
    testUVAll = []
    assert len(imgs)==len(uvs)

    for iMarker in range(len(imgs)):
        trainData, trainUV, testData, testUV, numData = divideTrainValSet(imgs[iMarker], uvs[iMarker], numTest, )
        numDataAll += numData
        sizeTrain = numData - numTest

        trainDataAll.append(trainData)
        trainUVAll.append(trainUV)
        testDataAll.append(testData)
        testUVAll.append(testUV)

    trainDataAll = np.concatenate(trainDataAll, axis=0)
    trainUVAll = np.concatenate(trainUVAll, axis=0)
    testDataAll = np.concatenate(testDataAll, axis=0)
    testUVAll = np.concatenate(testUVAll, axis=0)

    if shuffle:
        trainDataAll, trainUVAll = shuffleTogether(trainDataAll, trainUVAll)
        testDataAll, testUVAll = shuffleTogether(testDataAll, testUVAll)

    return trainDataAll, trainUVAll, testDataAll, testUVAll, numDataAll

def evaluate(imgs, groundTruth, uvExtractor, batchSize=10, perImgEval=True):
    predictions = uvExtractor.predict(imgs, batchSize=batchSize)
    evalStatistics = evaluatePrediction(predictions, groundTruth, perImgEval=perImgEval)

    return evalStatistics

def evaluatePrediction(predictions, groundtruth, perImgEval=True):
    diff = predictions - groundtruth
    errs = np.sqrt(diff[..., 0] **2 + diff[..., 1] **2)

    meanErr = np.mean(errs)
    maxErr = np.max(errs)
    medianErr = np.median(errs)
    err95Pctl = np.percentile(errs, 95)
    err99Pctl = np.percentile(errs, 99)
    err999Pctl = np.percentile(errs, 99.9)
    err9999Pctl = np.percentile(errs, 99.99)

    if perImgEval:
        perImgMeanErr = np.zeros((predictions.shape[0],))

        for iImg in range(predictions.shape[0]):
            perImgMeanErr[iImg] = np.mean(errs[iImg, ...])

        statistics = {
            'MeanErr':meanErr,
            'MedianErr': medianErr,
            'MaxErr': maxErr,
            'Error95Percentile': err95Pctl,
            'Error99Percentile': err99Pctl,
            'Error999Percentile': err999Pctl,
            'Error9999Percentile': err9999Pctl,
            'PerImageMeanErr':perImgMeanErr,
            'ErrorsAll': errs
        }
    else:
        statistics = {
            'MeanErr': meanErr,
            'MedianErr': medianErr,
            'MaxErr': maxErr,
            'Error95Percentile': err95Pctl,
            'Error99Percentile': err99Pctl,
            'Error999Percentile': err999Pctl,
            'Error9999Percentile': err9999Pctl,
        }

    return statistics

