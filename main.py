# Import the necessary packages
import numpy as np
import cv2
import fire
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Load the Waldo and puzzle images
waldo = cv2.imread('Images/wally2_1.jpg')
puzzle = cv2.imread('Images/wally02.jpg')

# Get the shape of Waldo's face image
(waldoHeight, waldoWidth) = waldo.shape[:2]

# Find Waldo in the puzzle
# cv2.TM_CCORR_NORMED is the cross-correlation normalized between the two images
result = cv2.matchTemplate(puzzle, waldo, cv2.TM_CCORR_NORMED)

# Set the threshold for find all the local max of the result
threshold = 0.9

# Find local max on the image where the pixel value was equal or bigger than the threshold
loc = np.where( result >= threshold)

# Get the Min and Max local of the cross-correlation normalized
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)

# Draw rectangles where is waldo
for pixel in zip(*loc[::-1]):
    cv2.rectangle(puzzle, pixel, (pixel[0] + waldoWidth, pixel[1] + waldoHeight), (0, 0, 0), 2)

# Display result matrix of cross-correlation in 3 dimensions
result_gray = result*255
xx, yy = np.mgrid[0:result_gray.shape[0], 0:result_gray.shape[1]]
fig = plt.figure(figsize=(15, 15))
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, result_gray, rstride=1, cstride=1, cmap='jet',linewidth=2)
ax.view_init(80, 30)
plt.show()


# Display the result of the Where is Waldo? problem
cv2.namedWindow('Where is Waldo', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Where is Waldo', 800, 600)
cv2.imshow('Where is Waldo', puzzle)
cv2.waitKey(0)
cv2.destroyAllWindows()

