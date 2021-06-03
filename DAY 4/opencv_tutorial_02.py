# USAGE
# python opencv_tutorial_02.py --image tetris_blocks.png

# import the necessary packages
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())

# load the input image (whose path was supplied via command line
# argument) and display the image to our screen
image = cv2.imread(args["image"])
cv2.imshow("Image", image)
cv2.waitKey(0)

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
cv2.waitKey(0)

# applying edge detection we can find the outlines of objects in
# images
edged = cv2.Canny(gray, 30, 150)
cv2.imshow("Edged", edged)
cv2.waitKey(0)

# Finding Contours
# Use a copy of the image e.g. edged.copy()
# since findContours alters the image
contours, hierarchy = cv2.findContours(edged,
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

print("Number of Contours found = " + str(len(contours)))

# draw the total number of contours found in purple
text = "I found {} objects!".format(len(contours))
cv2.putText(image, text, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (240, 0, 159), 2)
cv2.imshow("Contours", image)
cv2.waitKey(0)

# Draw all contours
# -1 signifies drawing all contours
imageCopy = image.copy()
cv2.drawContours(imageCopy, contours, -1, (0, 255, 0), 3)

cv2.imshow('Contours', imageCopy)
cv2.waitKey(0)

# threshold the image by setting all pixel values less than 225
# to 255 (white; foreground) and all pixel values >= 225 to 255
# (black; background), thereby segmenting the image
thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)

# we apply erosions to reduce the size of foreground objects
# None is the kernel

# Working of erosion:
# 1. A kernel(a matrix of odd size(3, 5, 7) is convolved with the image.
# 2. A pixel in the original image(either 1 or 0) will be considered 1 only if all the pixels under the #    kernel is 1, otherwise it is eroded(made to zero).
# 3. Thus all the pixels near boundary will be discarded depending upon the size of kernel.
# 4. So the thickness or size of the foreground object decreases or simply white region decreases in the #    image.

mask = thresh.copy()
mask = cv2.erode(mask, None, iterations=5)
cv2.imshow("Eroded", mask)
cv2.waitKey(0)

# similarly, dilations can increase the size of the ground objects
# Working of dilation:

# 1. A kernel(a matrix of odd size(3, 5, 7) is convolved with the image
# 2. A pixel element in the original image is ‘1’ if atleast one pixel under the kernel is ‘1’.
# 3. It increases the white region in the image or size of foreground object increases

mask = thresh.copy()
mask = cv2.dilate(mask, None, iterations=5)
cv2.imshow("Dilated", mask)
cv2.waitKey(0)

# a typical operation we may want to apply is to take our mask and
# apply a bitwise AND to our input image, keeping only the masked
# regions
mask = thresh.copy()
output = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Output", output)
cv2.waitKey(0)
