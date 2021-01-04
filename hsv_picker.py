import cv2
import numpy as np

image_hsv = None   # global ;(
pixel = (20,60,80) # some stupid default

cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
cv2.resizeWindow('mask', 480, 320)
cv2.namedWindow('bgr', cv2.WINDOW_NORMAL)
cv2.resizeWindow('bgr', 480, 320)
cv2.namedWindow('hsv', cv2.WINDOW_NORMAL)
cv2.resizeWindow('hsv', 960, 640)

# mouse callback function
def pick_color(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y,x]

        #you might want to adjust the ranges(+-10, etc):
        upper =  np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
        lower =  np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 40])
        print(pixel, lower, upper)

        image_mask = cv2.inRange(image_hsv,lower,upper)
        cv2.imshow("mask",image_mask)

def main():
    import sys
    global image_hsv, pixel # so we can use it in mouse callback

    image_src = cv2.imread(sys.argv[1])  # pick.py my.png
    if image_src is None:
        print ("the image read is None............")
        return
    cv2.imshow("bgr",image_src)

    ## NEW ##
    cv2.namedWindow('hsv')
    cv2.setMouseCallback('hsv', pick_color)

    # now click into the hsv img , and look at values:
    image_hsv = cv2.cvtColor(image_src,cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv",image_hsv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()


# import cv2

# img = cv2.imread("clips/frame.png")
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv,(10, 100, 20), (25, 255, 255) )
# cv2.imshow("orange", mask);cv2.waitKey();cv2.destroyAllWindows()

# import cv2
# import numpy as np


# def mouseRGB(event,x,y,flags,param):
#     if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
#         colorsB = image[y,x,0]
#         colorsG = image[y,x,1]
#         colorsR = image[y,x,2]
#         colors = image[y,x]
#         print("Red: ",colorsR)
#         print("Green: ",colorsG)
#         print("Blue: ",colorsB)
#         print("BRG Format: ",colors)
#         print("Coordinates of pixel: X: ",x,"Y: ",y)

# # Read an image, a window and bind the function to window
# image = cv2.imread("./clips/frame.png")
# cv2.namedWindow('mouseRGB', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('mouseRGB', 960, 640)
# cv2.setMouseCallback('mouseRGB',mouseRGB)

# #Do until esc pressed
# while(1):
#     cv2.imshow('mouseRGB',image)
#     if cv2.waitKey(20) & 0xFF == 27:
#         break
# #if esc pressed, finish.
# cv2.destroyAllWindows()