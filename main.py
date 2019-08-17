import cv2


DIR = '/home/tomas/Documents/Universidad/DIP/HW 1/Where_is_Waldo/vwru869.jpg'

image = cv2.imread(DIR)

cv2.namedWindow('Where is Waldo', cv2.WINDOW_NORMAL)
cv2.imshow('Where is Waldo', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
