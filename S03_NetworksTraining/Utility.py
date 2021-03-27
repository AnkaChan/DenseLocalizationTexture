import numpy as np
import cv2, json
from os.path import join
import random
from matplotlib import pyplot as plt

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


def plotPredictionComparison(imgs, imgIdsToVis, predictions, uvs, evalStatistics, plotName=''):
    gridH = len(imgIdsToVis)
    gridW = 4

    fig, axs = plt.subplots(gridH, gridW)

    fig.set_size_inches(20, 20 * (gridH / gridW))
    fig.suptitle('Prediction visualization of: ' + plotName)

    for i, iImg in enumerate(imgIdsToVis):
        img = imgs[iImg, ...]
        pred = predictions[iImg, ...]
        gd = uvs[iImg, ...]
        errs = evalStatistics['ErrorsAll'][iImg, ...]

        predRGB, gdRGB = predAndGdToRGB(pred, gd, normalize=True)

        axs[i, 0].imshow(img)
        axs[i, 0].set_title("img: %d" % (iImg,))
        axs[i, 0].axis('off')

        axs[i, 1].imshow(predRGB)
        axs[i, 1].axis('off')

        axs[i, 2].imshow(gdRGB)
        axs[i, 2].axis('off')

        pos = axs[i, 3].imshow(errs, cmap='jet')
        #         axs[i, 3].legend()
        axs[i, 3].axis('off')
        fig.colorbar(pos, ax=axs[i, 3])
        axs[i, 3].set_title("Mean Err: %f" % (evalStatistics['PerImageMeanErr'][iImg]))

    return fig


def visualizeEvaluationOfUVExtractor(imgs, uvs, uvExtractor, imgsToShow='maxErr', numImgs=10, showFigure=True,
                                     saveFile=None):
    plt.close('all')

    predictions = uvExtractor.predict(imgs, batchSize=20)

    evalStatistics = evaluatePrediction(predictions, uvs)

    errStr = []
    for k, v in evalStatistics.items():
        if k != 'PerImageMeanErr' and k != 'ErrorsAll':
            # print(k, v)
            errStr.append(k + ': ' + str(v))
    errStr = '\n'.join(errStr)

    n_bins = 50
    # We can set the number of bins with the `bins` kwarg
    figHist = plt.figure()
    kwargs = dict(alpha=0.5, stacked=True)

    # linear
    allErrs = evalStatistics['ErrorsAll'].flatten()
    hist, bins = np.histogram(allErrs, bins=n_bins)
    bins = np.logspace(np.log10(bins[0] + 1e-4), np.log10(bins[-1]), len(bins))

    # # linear
    # kwargs = dict(alpha=0.5, bins=400, stacked=True)
    # # fig = plt.figure(constrained_layout=True, tight_layout=True)
    # plt.rcParams["figure.figsize"] = (6, 2.5)  # (w, h)
    # fig = plt.figure(tight_layout=True)
    # # gs = GridSpec(1, 1, figure=fig)

    # linear
    plt.hist(allErrs, **kwargs, color='b', bins=bins)
    plt.gca().set(ylabel='Bin count', xlabel='Prediction Errors')

    # plt.yscale('log')
    plt.xscale('log')

    # Visualize 10 frames with highest prediction error

    cmprPlots = []
    if imgsToShow == 'maxErr':
        imgIdsHighestMeanErrs = np.argsort(evalStatistics['PerImageMeanErr'])[-1:-1 - numImgs:-1]
        figCmprHighestMeanErrs = plotPredictionComparison(imgs, imgIdsHighestMeanErrs, predictions, uvs, evalStatistics,
                                                          plotName=str(numImgs) + ' images with highest mean error')
        cmprPlots.append(figCmprHighestMeanErrs)
    elif imgsToShow == 'random':
        imgIdsRandom = np.random.choice(evalStatistics['PerImageMeanErr'].shape[0], numImgs)
        figCmprRandom = plotPredictionComparison(imgs, imgIdsRandom, predictions, uvs, evalStatistics,
                                                 plotName=str(numImgs) + ' randomly picked images')
        cmprPlots.append(figCmprRandom)

    elif imgsToShow == 'both':
        imgIdsHighestMeanErrs = np.argsort(evalStatistics['PerImageMeanErr'])[-1:-1 - numImgs:-1]
        figCmprHighestMeanErrs = plotPredictionComparison(imgs, imgIdsHighestMeanErrs, predictions, uvs, evalStatistics,
                                                          plotName=str(numImgs) + ' images with highest mean error')
        cmprPlots.append(figCmprHighestMeanErrs)

        imgIdsRandom = np.random.choice(evalStatistics['PerImageMeanErr'].shape[0], numImgs)
        figCmprRandom = plotPredictionComparison(imgs, imgIdsRandom, predictions, uvs, evalStatistics,
                                                 plotName=str(numImgs) + ' randomly picked images')
        cmprPlots.append(figCmprRandom)

    if showFigure:
        plt.show()

    if saveFile is not None:
        ax, figErrs = plt.subplots()
        ax.text(0.5, 0.5, errStr, ha='center', va='center', )

        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(saveFile) as pdf:
            pdf.savefig(ax)
            # pdf.save

            pdf.savefig(figHist)

            for fig in cmprPlots:
                pdf.savefig(fig)


