import argparse
import json
import cv2
import os
import torch
import torch_directml
import time
import numpy as np
from torchvision import models,transforms

import torch.nn as nn

from torchvision.models import resnet50
from torch.nn.functional import normalize
from tqdm import tqdm
from scipy.spatial.distance import cosine
from cat_info import CatInfo

baseFolder = "./images/"

print("Cat recognition system")

useModelWithReg = True
useColorNormalization = True

dbFileName = baseFolder + "catsDB" 
if useColorNormalization:
    dbFileName += "_wCN"
if useModelWithReg:
    dbFileName += "_wReg"
dbFileName += ".json"


if not os.path.isfile(dbFileName):
    print("Cat database file not found, please run CatDatabaseBuilder.py first")
    exit()

#load cats from file
catInfoArray = []
with open(dbFileName) as json_file:
    jsonArr = json.load(json_file)
    for jsonObj in jsonArr:
        catInfoArray.append(CatInfo.FromJson(jsonObj))
        #print short description and tensor size
        print (catInfoArray[-1].ShortDescription(), " length ", len(catInfoArray[-1].outputTensor))
    print("Loaded ", len(jsonArr), " cats from file")

#define compute cosine distance between two catInfo objects
def computeDistance(catInfo1, catInfo2):
    return cosine(catInfo1.outputTensor, catInfo2.outputTensor)

def printDistance(catInfo1, catInfo2):
    print("Distance between ", catInfo1.ShortDescription(), " and ", catInfo2.ShortDescription(), " is ", computeDistance(catInfo1, catInfo2))

#order array by catName
catInfoArray.sort(key=lambda x: x.catName, reverse=False)
#filter out night vision images
catInfoArray = [x for x in catInfoArray if x.nightVision == False]

#create table with computeDistance between all cats
distanceTable = np.zeros((len(catInfoArray), len(catInfoArray)))
for i in range(len(catInfoArray)):
    for j in range(len(catInfoArray)):
        distanceTable[i][j] = computeDistance(catInfoArray[i], catInfoArray[j])

#print distance table formatted to command line
print("Distance table:")
print("      ", end="")
for catInfo in catInfoArray:
    print("{:9s}".format(catInfo.ShortDescription()), end="")
print("")
for i in range(len(catInfoArray)):
    print("{:9s}".format(catInfoArray[i].ShortDescription()), end="")
    for j in range(len(catInfoArray)):
        #print value in color from green to red depending on distance
        distance = distanceTable[i][j]
        #compute color based on distance. 0 is green, 1 is red
        vcc = int(127 * (distance / 1))
        if vcc > 127:
            vcc = 127
        if vcc < 0:
            vcc = 0
        color = (127+vcc, 127-vcc, 127)
        #print("{:12.2f}".format(), end="")
        print("\033[38;2;{};{};{}m{:9.2f}\033[0m".format(color[2], color[1], color[0], distance), end="")
    print("")
print("")
print("")


#use GPU to speed up
dmlDevice = torch_directml.device()
print("torch version: ", torch.__version__)
print(f'Using device {dmlDevice} for inference')

#load medium size dinov2 model
#load medium size dinov2 model
if useModelWithReg:
    dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
else:
    dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2_vitb14_reg.eval().to(dmlDevice)
from torchvision import models,transforms

# Define the transformation
preprocess = transforms.Compose([
    transforms.ToTensor()
])


preprocessWCN = transforms.Compose([
    # convert the frame to a CHW torch tensor for training
    transforms.ToTensor(),
    # normalize the colors to the range that mobilenet_v2/3 expect
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def GetTensorForUnknownImage(im):
    #if dimension of image is not devidable by 14, resize it
    scale = 0.2
    newWidth = int(im.shape[1]*scale/14)*14        
    newHeight = int(im.shape[0]*scale/14)*14
    newDim = (newWidth, newHeight)
    scaledim = cv2.resize(im, newDim, interpolation = cv2.INTER_AREA)
    #transform image to tensor and copy to GPU
    if useColorNormalization:
        imArray = [
            preprocessWCN(scaledim).unsqueeze(0)
        ]
    else:
        imArray = [
            preprocess(scaledim).unsqueeze(0)
        ]
    modelInputBatch = torch.cat(imArray).to(dmlDevice)
    #compute output
    modelOutput = dinov2_vitb14_reg(modelInputBatch).cpu()
    #return tensor
    return modelOutput[0].detach().numpy()


def compareUnknownImageFileToDB(fileName):
    unknownTensor = GetTensorForUnknownImage(cv2.imread(fileName))
    #prin tensor length
    print ("tensor length: ", len(unknownTensor))
    for i in range(0,catInfoArray.__len__()):
        tensorDist = cosine(unknownTensor, catInfoArray[i].outputTensor)
        print ("distance " + fileName + "-" + catInfoArray[i].ShortDescription() + ": " , tensorDist)

catNames = ["Amalka", "Gita", "Kocour", "Maugli"]



print ("compare to unknown images")
#compareUnknownImageFileToDB("./images/maugli1.jpg")
#compareUnknownImageFileToDB("./images/maugli2.jpg")
#compareUnknownImageFileToDB("./images/maugli3.jpg")
#compareUnknownImageFileToDB("./images/gita1.jpg")
#compareUnknownImageFileToDB("./images/gita2.jpg")


#haar detector
detector = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")

videoFilePath = "D:/temp/cam-kocky/20231125_130743_tp00001.mp4"
cv2VideoF = cv2.VideoCapture(videoFilePath)
timeInVideo = "18:00"#mm:ss
timeMs = int(timeInVideo.split(":")[0]) * 60 * 1000 + int(timeInVideo.split(":")[1]) * 1000
firstFrame = int(timeInVideo.split(":")[0]) * 60  + int(timeInVideo.split(":")[1])
firstFrame = firstFrame*15
cv2VideoF.set(cv2.CAP_PROP_POS_MSEC, timeMs)
#read 10 frames and call compareVideoFrameToDB

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
outputVideoWrite = cv2.VideoWriter('output.mp4', fourcc, 15.0, (1280,720))


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


def InsertDetectedCatsIntoImage(im, rects):
    for (i, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(im, "Cat #{}".format(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

startTime = time.time()
frameCount = 500


catCounterFIRfilter = [0, 0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0]
lastNotificationTime = time.time() - 60

for i in range(frameCount):
    ret, frame = cv2VideoF.read()
    if ret:
        scale = 0.33
        newWidth = int(frame.shape[1])        
        newHeight = int(frame.shape[0])
        newDim = (newWidth, newHeight)
        gray = cv2.cvtColor(cv2.resize(frame, newDim, interpolation = cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        rects = detector.detectMultiScale(gray, scaleFactor=1.3,minNeighbors=10, minSize=(75, 75))
        catCount = rects.__len__()

        #FIR filter
        catCounterFIRfilter.pop(0)
        if (catCount > 0):
            catCounterFIRfilter.append(1)
        else:
            catCounterFIRfilter.append(0)
        filterOutValue = sum(catCounterFIRfilter)/len(catCounterFIRfilter)

        if (catCount > 0) or (filterOutValue > 0.05):
            compareVideoFrameToDB(frame,i)
        if (catCount > 0):
            InsertDetectedCatsIntoImage(frame, rects)
        #cv2.imshow("Cat video", frame)
        #cv2.waitKey(1)
        outputVideoWrite.write(frame)

    else:
        print("end of video")
        break
endTime = time.time()
print ("GPU processing time (" + str(frameCount) + " images): ", endTime - startTime)
#print fps
print ("fps: ", frameCount / (endTime - startTime))

cv2VideoF.release()
outputVideoWrite.release()