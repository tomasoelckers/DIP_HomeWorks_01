from scipy import signal
import scipy as ssc
import numpy as np
import cv2
import tools

DIR = '/home/tomas/Documents/Universidad/DIP/HW 1/Where_is_Waldo/vwru869.jpg'

image1 = cv2.imread('Images/wally2_1.jpg', 0)
image2 = cv2.imread('Images/wally02.jpg', 0)

cv2.namedWindow('Where is Waldo', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Where is Waldo', 800,600)
cv2.imshow('Where is Waldo', image1)

print('Images Imported')

result = tools.convolution(image2, image1)

print('Correlated Images ')

cv2.imshow('Where is Waldo', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

