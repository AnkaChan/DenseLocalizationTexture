import cv2
import OpenEXR
import Imath
import sys
import array
import numpy as np
# def read_depth_exr_file(filepath: Path):
#     exrfile = exr.InputFile(filepath.as_posix())
#     raw_bytes = exrfile.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
#     depth_vector = numpy.frombuffer(raw_bytes, dtype=numpy.float32)
#     height = exrfile.header()['displayWindow'].max.y + 1 - exrfile.header()['displayWindow'].min.y
#     width = exrfile.header()['displayWindow'].max.x + 1 - exrfile.header()['displayWindow'].min.x
#     depth_map = numpy.reshape(depth_vector, (height, width))
#     return depth_map

if __name__ == '__main__':
    inFile = r'TestRendering.exr'

    # img = cv2.imread(inFile,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    # print(img.shape)

    # if len(sys.argv) != 3:
    #     print("usage: exrnormalize.py exr-input-file exr-output-file")
    #     sys.exit(1)

    # Open the input file
    file = OpenEXR.InputFile(inFile)

    # Compute the size
    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R, G, B, Z) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B", "Z")]
    # (R, G, B, Z) = [array.array('f', file.channel(Chan, FLOAT)) for Chan in ("R", "G", "B", "Z")]

    r = np.array(R)
    Z = np.array(Z)
    r.resize(sz[1], sz[0])
    Z.resize(sz[1], sz[0])
    Z = np.clip(Z, 0, 30)
    print(Z[500, 1000])
    print(Z[1, 1])
    cv2.imshow('r',r)
    cv2.imshow('Z',Z/np.max(Z))
    cv2.waitKey()

    # print(Z)