import cv2
import numpy as np

MAX_FRAMES = 10
LEARNING_RATE = -1   
#fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg = cv2.createBackgroundSubtractorKNN()

#image should be grayscale
def GetMotionMask(image):    
    blur = cv2.GaussianBlur(image, (7, 7), 0) 
    #cv2.imshow('frame', frame)
    # black out polygon area with time and non interesting area
    pts = np.array([[0, 0], [0, 360], [80, 360], [150, 18] , [227, 18], [227, 0]], np.int32)
    cv2.fillPoly(blur, [pts], (0))
    #Apply MOG 
    motion_mask = fgbg.apply(blur, LEARNING_RATE)
    #cv2.imshow('motionMask', motion_mask)
    thresh_frame = cv2.threshold(motion_mask, 30, 255, cv2.THRESH_BINARY)[1] 
    return thresh_frame

motionDetected = False

def putImageIntoProcessPipeline(image):
    global motionDetected
    mm = GetMotionMask(image)
    nonZero = np.count_nonzero(mm)
    if (nonZero > 500):
        motionDetected = True
        print("motion detected, nonZero pixels: ", nonZero)
        #cv2.imshow('motion detected', image)
        #cv2.waitKey(1)

