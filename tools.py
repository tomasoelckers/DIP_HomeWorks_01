import numpy as np
import cv2


def convolution(image,kernel):
    buff = image.copy()
    height, width = buff.shape # 1080 + 2, 1920 + 2
    H, W = kernel.shape
    buff = np.pad(buff, pad_width=H, mode='constant', constant_values=0)
    for h in range(height):
        h += H
        for w in range(width):
            w += W
            nb = list()
            for i in range(H):
                for j in range(W):
                    nb.append((buff[h+i][w+j])*(kernel[i][j]))
            buff[h][w] = sum(nb)

    return buff


def cross_convolution_2d():
    pass

