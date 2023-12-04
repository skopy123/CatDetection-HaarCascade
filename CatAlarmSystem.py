
#system
import argparse
import json
import cv2
import os
import time
import traceback

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
import modMotionDetection as motionDetection
import modImageTools as imageTools
import modHaarCascade as catHaarCascade

print("Cat recognition system")

aiProcessedFrames = 0
aiProcessingTime = 0

def processFrame(im:cv2.typing.MatLike, frameNumber:int):
    if (frameNumber % 10 > 0):
        return
    print("processing frame: ", frameNumber)
    global aiProcessedFrames
    global aiProcessingTime
    #stage1 getGrayScaleImage
    gray, nvMode = imageTools.ConvertToGrayScaleIfNecessary(im)

    #stage1 - motion detection
    motionDetection.putImageIntoProcessPipeline(gray)
    if (motionDetection.motionDetected == False):
        return #no motion detected, skip rest of the processing
    
    #stage2 haar cascade detection
    #catHaarCascade.PutImageIntoProcessPipeline(gray)
    #if (catHaarCascade.catPresenceFiltred == False):
        #return
       #print("movement detected bud haar cascade(filtrated) do not find cat")

    #stage3 compare to DB
    aiBeginTime = time.time()
    scaledImagae = image2Tensor.ResizeImageToModelInput(im,14*14)
    # cut left 50 pixels from image
    scaledImagae = scaledImagae[:,42:,:]
    #print scaled image size
    height, width, channels = scaledImagae.shape
    print (f"procssing image, dimension:{height}x{width}")
    unknownTensor = image2Tensor.ComputeTensorForImage(scaledImagae)
    catMatches = templateDB.CompareUnknownTensorToDB(unknownTensor,nvMode)

    #stage4 enhance image
    templateDB.AddStatsIntoImage(im, catMatches)
    if (catHaarCascade.lastImageCatCount > 0):
        catHaarCascade.AddBoundingBoxesIntoImage(im)
    #show image
    cv2.imshow("Cat video", frame)
    cv2.waitKey(1)

    aiEndTime = time.time()
    aiProcessingTime = aiProcessingTime + (aiEndTime - aiBeginTime)
    aiProcessedFrames = aiProcessedFrames + 1
    #eval results
    if (catMatches[0].DistanceToTemplatesMin < catMatches[0].DistanceToTemplatesMin):
        print("cat detected: ", catMatches[0].Name, "dist: ", catMatches[0].DistanceToTemplatesMin)
        if (catMatches[0].Name == "Maugli"):
            pb.SendNotificationWithRateLimiter("Cat alarm", "Cat detected: " + catMatches[0].Name + " dist:" + f"{catMatches[0].DistanceToTemplatesMin:.2f}")

    #outputVideoWrite.write(frame)
#output video stream
#outputVideoWrite = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (1280,720))

#stream

netstream = cv2.VideoCapture(os.getenv("RSTP_STREAM"))
startT = time.time()
frameCounter  = 0
while (1):
    try:
        ret, frame = netstream.read()
        if not ret:
            print("Frame is empty")
            time.sleep(0.1)
            continue
        
        while True:
            ret_new, frame_new = netstream.read()
            if not ret_new:
                break
            else:
                frame = frame_new

        frameCounter =  frameCounter + 1
            
        processFrame(frame, frameCounter)
            
        if (frameCounter % 100 == 0):   
            ensT = time.time()
            print("total avg FPS: ", frameCounter/(ensT - startT))
            print("AI avg FPS: ", aiProcessedFrames/aiProcessingTime)

    except Exception as e:
        print("Error in processFrame")
        print(str(e))  # Print the error message
        traceback.print_exc()  # Print the full traceback
        time.sleep(5)


#videoFilePath = "D:/temp/cam-kocky/20231125_130743_tp00001.mp4"
#cv2VideoF = cv2.VideoCapture(videoFilePath)
#timeInVideo = "18:00"#mm:ss
#timeMs = int(timeInVideo.split(":")[0]) * 60 * 1000 + int(timeInVideo.split(":")[1]) * 1000
#firstFrame = int(timeInVideo.split(":")[0]) * 60  + int(timeInVideo.split(":")[1])
#firstFrame = firstFrame*15
#cv2VideoF.set(cv2.CAP_PROP_POS_MSEC, timeMs)
#read 10 frames and call compareVideoFrameToDB
