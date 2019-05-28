# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 09:44:41 2018

@author: V SUSHANT
"""

import cv2
import numpy as np

# load image
img = cv2.imread('single_box.jpg') 

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

# threshold to get just the signature
retval, thresh_gray = cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_BINARY)

# find where the black pixels are
points = np.argwhere(thresh_gray==0) 

# store them in x,y coordinates instead of row,col indices
points = np.fliplr(points) 

# create a rectangle around those points
x, y, w, h = cv2.boundingRect(points)

# make the box a little bigger 
x, y, w, h = x+1, y+1, w-2, h-2

# create a cropped region of the gray image 
crop = gray[y:y+h, x:x+w] 

# get the thresholded crop
retval, thresh_crop = cv2.threshold(crop, thresh=200, maxval=255, type=cv2.THRESH_BINARY)

# display
cv2.imshow("Cropped and thresholded image", thresh_crop) 
cv2.waitKey(0)
cv2.imwrite("edge.jpg",thresh_crop)