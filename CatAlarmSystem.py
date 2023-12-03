
#system
import argparse
import json
import cv2
import os
import time

#vision and ML
import torch
#import torch_directml

import numpy as np
from torchvision import models,transforms
import torch.nn as nn
from torch.nn.functional import normalize
from tqdm import tqdm
from scipy.spatial.distance import cosine


#my modules
from catMatchStats import CatMatchStats
from cat_info import CatInfo
import modTemplateDatabase as templateDB
import modImage2Tensor as image2Tensor
import modSettings as settings
import modPushBulletWrapper as pb

print("Cat recognition system")

def GetTensorForUnknownImage(im):
    #if dimension of image is not devidable by 14, resize it
    scale = 0.2
    newWidth = int(im.shape[1]*scale/14)*14        
    newHeight = int(im.shape[0]*scale/14)*14
    newDim = (newWidth, newHeight)
    scaledim = cv2.resize(im, newDim, interpolation = cv2.INTER_AREA)
    #transform image to tensor and copy to GPU
   
    return image2Tensor.computeTensorForImage(scaledim)

def compareUnknownImageFileToDB(fileName):
    unknownTensor = GetTensorForUnknownImage(cv2.imread(fileName))
    #prin tensor length
    print ("tensor length: ", len(unknownTensor))
    for i in range(0,catInfoArray.__len__()):
        tensorDist = cosine(unknownTensor, catInfoArray[i].outputTensor)
        print ("distance " + fileName + "-" + catInfoArray[i].ShortDescription() + ": " , tensorDist)

catNames = ["Amalka", "Gita", "Kocour", "Maugli"]





lastNotificationTime = time.time() - 60

def compareVideoFrameToDB(im, frameNumber):
    unknownTensor = GetTensorForUnknownImage(im)
    #prin tensor length
    minDist = 1
    minDistIndex = -1
    minDistPerCat = [1, 1, 1, 1]
    totalDistPerCat = [0, 0, 0, 0]
    countPerCat = [0, 0, 0, 0]
    for i in range(0,catInfoArray.__len__()):
        tensorDist = cosine(unknownTensor, catInfoArray[i].outputTensor)
        catIndex = catNames.index(catInfoArray[i].catName)
        totalDistPerCat[catIndex] += tensorDist
        countPerCat[catIndex] += 1
        if tensorDist < minDistPerCat[catIndex]:
            minDistPerCat[catIndex] = tensorDist
        if tensorDist < minDist:
            minDist = tensorDist
            minDistIndex = i
    if minDistIndex >= 0:
        print ("frame" + str(frameNumber) + ",  minimal distance " + catInfoArray[minDistIndex].ShortDescription() + ": " , f"{minDist:.2f}")
    #add text to low left corner
    for i in range(0,4):
        if countPerCat[i] == 0: continue
        #format avg and min distance to 2 decimal places
        text = catNames[i] + " avg:" + f"{totalDistPerCat[i]/countPerCat[i]:.2f}" + " min:" + f"{minDistPerCat[i]:.2f}"
        pos = (20, im.shape[0] - 20 - 20 * (i+1))
        cv2.putText(im, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    pos = (20, im.shape[0] - 20)
    text = "closest match:" +catInfoArray[minDistIndex].ShortDescription() + " dist:" + f"{minDist:.2f}"
    cv2.putText(im, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    #send notification if distance to maugli is less than 0.5

    global lastNotificationTime
  #  if (minDist < 0.3) and (catInfoArray[minDistIndex].catName == "maugli") and ((time.time() - lastNotificationTime) > 60):
    if (minDist < 0.3) and ((time.time() - lastNotificationTime) > 60):
        lastNotificationTime = time.time()
        pushBulletApi.send_note("Cat alarm", "Cat detected: " + catInfoArray[minDistIndex].catName + " dist:" + f"{minDist:.2f}")


def InsertDetectedCatsIntoImage(im, rects):
    for (i, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(im, "Cat #{}".format(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)






def processFrame(im, frameNumber):
    scale = 0.33
    newWidth = int(frame.shape[1])        
    newHeight = int(frame.shape[0])
    newDim = (newWidth, newHeight)
    gray = cv2.cvtColor(cv2.resize(frame, newDim, interpolation = cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    #rects = detector.detectMultiScale(gray, scaleFactor=1.3,minNeighbors=10, minSize=(75, 75))

    if (catCount > 0):
        #InsertDetectedCatsIntoImage(frame, rects)
        cv2.imshow("Cat video", frame)
        cv2.waitKey(1)
    #outputVideoWrite.write(frame)
#output video stream
#outputVideoWrite = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (1280,720))

#stream
netstream = cv2.VideoCapture('rtsp://maugli:vetrelec1@192.168.129.198:554/stream1')
startT = time.time()
frameCounter  = 0
while (1):
    ret, frame = netstream.read()
    if ret == False:
        print("Frame is empty")
        break
    else:
        processFrame(frame, i)
        #cv2.imshow('liveStream', frame)
        #cv2.waitKey(1)
        frameCounter += 1
        if (frameCounter % 100 == 0):
            ensT = time.time()
            print("FPS: ", frameCounter/(ensT - startT))
            frameCount = 0
            startT = time.time()

#outputVideoWrite.release()
exit()



videoFilePath = "D:/temp/cam-kocky/20231125_130743_tp00001.mp4"
cv2VideoF = cv2.VideoCapture(videoFilePath)
timeInVideo = "18:00"#mm:ss
timeMs = int(timeInVideo.split(":")[0]) * 60 * 1000 + int(timeInVideo.split(":")[1]) * 1000
firstFrame = int(timeInVideo.split(":")[0]) * 60  + int(timeInVideo.split(":")[1])
firstFrame = firstFrame*15
cv2VideoF.set(cv2.CAP_PROP_POS_MSEC, timeMs)
#read 10 frames and call compareVideoFrameToDB

startTime = time.time()
frameCount = 2500
for i in range(frameCount):
    ret, frame = cv2VideoF.read()
    if ret:
        processFrame(frame, i) 
    else:
        print("end of video")
        break
endTime = time.time()
print ("GPU processing time (" + str(frameCount) + " images): ", endTime - startTime)
#print fps
print ("fps: ", frameCount / (endTime - startTime))

cv2VideoF.release()
outputVideoWrite.release()