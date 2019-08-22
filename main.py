# Import the necessary packages
import numpy as np
import cv2
import fire
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# The name of the image file
DIR = 'wally2'

# Load the Waldo and puzzle images
# Waldo's picture direction path
waldo = cv2.imread('Images/wally2_1.jpg')
# Puzzle image direction path
puzzle = cv2.imread('Images/'+ DIR + '.jpg')

# Get the shape of Waldo's face image
(waldoHeight, waldoWidth) = waldo.shape[:2]

# Find Waldo in the puzzle
# cv2.TM_CCORR_NORMED is the cross-correlation normalized between the two images
result = cv2.matchTemplate(puzzle, waldo, cv2.TM_CCOEFF_NORMED)

# Set the threshold for find all the local max of the result
threshold = 0.9

# Find local max on the image where the pixel value was equal or bigger than the threshold
loc = np.where( result >= threshold)

# Get the Min and Max local of the cross-correlation normalized
#(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)

# Draw rectangles where is waldo
for pixel in zip(*loc[::-1]):
    cv2.rectangle(puzzle, pixel, (pixel[0] + waldoWidth, pixel[1] + waldoHeight), (0, 0, 0), 2)

# Saving image
#cv2.imwrite('Result Images/' + DIR + '_1.jpg', puzzle)

# Display the result of the Where is Waldo? problem
'''plt.figure(1)
plt.imshow(cv2.cvtColor(puzzle, cv2.COLOR_BGR2RGB))
plt.show()'''


# Display result matrix of cross-correlation in 3 dimensions
plt.figure(2)
result_gray = result*255
xx, yy = np.mgrid[0:result_gray.shape[0], 0:result_gray.shape[1]]
fig = plt.figure(figsize=(15, 15))
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, result_gray, rstride=1, cstride=1, cmap='jet', linewidth=2, antialiased=False)
ax.view_init(60, 30)
plt.show()


# Display the result of the Where is Waldo? problem
cv2.namedWindow('Where is Waldo', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Where is Waldo', 800, 600)
cv2.imshow('Where is Waldo', puzzle)
cv2.waitKey(0)
cv2.destroyAllWindows()
