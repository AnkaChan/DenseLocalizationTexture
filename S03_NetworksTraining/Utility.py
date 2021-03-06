import numpy as np
import cv2

def predToRGB(outputs):
    return np.stack([outputs[..., 0], outputs[..., 1], np.zeros(outputs[..., 1].shape)], -1)
