import numpy as np
import cv2

if __name__ == '__main__':
    W = 2000
    H = 2000
    m = 20
    minSize = 2

    outFile = r"RandGNoise_NN_2000x2000_Coarse.bmp"
    nChannels = 1

    interpoMethod = cv2.INTER_CUBIC
    interpoMethod = cv2.INTER_NEAREST

    img = np.zeros((H, W, nChannels))
    np.random.seed(12345)

    for iC in range(nChannels):
        count = 0
        pattern = np.zeros((H, W))
        while m < W / minSize:
            n = round(H / W * m)
            noise = np.random.uniform(size=(n, m))
            # dim = (width, height)
            noise = cv2.resize(noise, (W, H), interpolation=interpoMethod)
            pattern = pattern + noise
            count = count + 1
            m = m * 2

        pattern = pattern / count
        img[:, :, iC] = 255 * pattern
        img[:, :, iC] = cv2.equalizeHist(img[:, :, iC].astype(np.uint8))

    img = np.squeeze(img)
    # img = cv2.equalizeHist(img.astype(np.uint8))

    cv2.imwrite(outFile, img)
    cv2.imshow("texture", img)
    cv2.waitKey()