import argparse
import json
import cv2
import os
import torch
#import torch_directml
import time
import numpy as np
from torchvision import models,transforms

import torch.nn as nn

#from torchvision.models import resnet50
from torch.nn.functional import normalize
from tqdm import tqdm
from scipy.spatial.distance import cosine
from cat_info import CatInfo
#from pushbullet import API as pbApi

videoFilePath = "D:/temp/cam-kocky/20231125_130743_tp00001.mp4"
#netstream = cv2.VideoCapture(videoFilePath)

netstream = cv2.VideoCapture('rtsp://maugli:vetrelec1@192.168.129.198:554/stream2')

MAX_FRAMES = 10
LEARNING_RATE = -1   
#fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg = cv2.createBackgroundSubtractorKNN()

#cap = cv2.VideoCapture(0)

def DetectNigthVision(image):
    ### splitting b,g,r channels
    b,g,r=cv2.split(image)

    ### getting differences between (b,g), (r,g), (b,r) channel pixels
    r_g=np.count_nonzero(abs(r-g))
    r_b=np.count_nonzero(abs(r-b))
    g_b=np.count_nonzero(abs(g-b))

    ### sum of differences
    diff_sum=float(r_g+r_b+g_b)

    ### finding ratio of diff_sum with respect to size of image
    ratio=diff_sum/image.size

    print("ratio: ", ratio)
    if ratio>0.1:
        print("image is color")
    else:
        print("image is greyscale")

startT = time.time()
frameCount = 0

while (1):
    ret, frame = netstream.read()
    if ret == False:
        print("Frame is empty wait for 5s")
        time.sleep(5)
    else:

        frameCount += 1
        if (frameCount % 5 > 0):
            continue
        #convert to gray, even night vision images to save processing time
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.GaussianBlur(gray, (7, 7), 0) 

    #cv2.imshow('frame', frame)
    # black out polygon area
        pts = np.array([[0, 0], [0, 360], [80, 360], [150, 18] , [227, 18], [227, 0]], np.int32)
        cv2.fillPoly(gray, [pts], (0))
        #cv2.imshow('gray', gray)
    #Apply MOG 
        motion_mask = fgbg.apply(gray, LEARNING_RATE)
        #cv2.imshow('motionMask', motion_mask)
        thresh_frame = cv2.threshold(motion_mask, 30, 255, cv2.THRESH_BINARY)[1] 
        #cv2.imshow('thresh_frame', thresh_frame)
        #cv2.waitKey(1)
        nonZero = np.count_nonzero(thresh_frame)
        if (nonZero > 500):
            print("nonZero: ", nonZero)
            cv2.imshow('motion detected', frame)
            cv2.waitKey(1)
       # time.sleep(0.3)
    if (frameCount % 100 == 0):
        ensT = time.time()
        print("FPS: ", frameCount/(ensT - startT))
        frameCount = 0
        startT = time.time()