from DifferenceImagesUsingSubtract import Subtract
from DifferenceImagesUsingABSDiff import ABSDiff
from DifferenceImagesUsingCompareSSIM import CompareSSIM
import cv2

if __name__ == '__main__':
    image1 = "image1.jpg"
    image2 = "image2.jpg"

    #diff1, diff2 = Subtract(image1, image2)
    #diff1, diff2 = CompareSSIM(image1, image2)
    diff1, diff2 = ABSDiff(image1, image2)

    #ekranda gosterme
    cv2.imshow('Image1', diff1)
    cv2.imshow('Image2', diff2)

    cv2.waitKey(0)



