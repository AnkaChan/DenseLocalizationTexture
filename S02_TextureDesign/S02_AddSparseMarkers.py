import cv2
import itertools
if __name__ == '__main__':
    # 50x50 markers, 200x200 grids

    inputTexture = r"RandGNoise_NN_2000x2000_Coarse.bmp"
    outputTexture = r"RandGNoise_NN_2000x2000_WithMarkers.bmp"

    gridSize = 20
    gridEdgeLen = 100
    markerColor = 0
    markerColor2 = 255

    markerWidth = 6
    markerLength = 50
    hMarkerW = int(markerWidth/2)
    hMarkerL = int(markerLength/2)

    texture = cv2.imread(inputTexture, )

    for i, j in itertools.product(range(1, gridSize), range(1, gridSize)):
        texture[i*gridEdgeLen-hMarkerW:i*gridEdgeLen+hMarkerW, j*gridEdgeLen-hMarkerL:j*gridEdgeLen+hMarkerL] = markerColor
        texture[i*gridEdgeLen:i*gridEdgeLen+hMarkerL, j*gridEdgeLen-hMarkerW:j*gridEdgeLen+hMarkerW] = markerColor
        texture[i*gridEdgeLen-hMarkerL:i*gridEdgeLen, j*gridEdgeLen-hMarkerW:j*gridEdgeLen+hMarkerW] = markerColor2

    # cv2.imshow('TextureWithMarker', texture)
    # cv2.waitKey()

    cv2.imwrite(outputTexture, texture)