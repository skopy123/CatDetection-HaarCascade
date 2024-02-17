
#system
import argparse
import json
import cv2
import os
import time
import traceback
import threading
import queue

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
import modhaHttpSensorPush as haApi

print("Cat recognition system")

aiProcessedFrames = 0
aiProcessingTime = 0

stop_threads = False

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
        haApi.HaPostSensorValueWithRateLimiter("noMotion", "http://hassio.lan:8124")
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
    cv2.imshow("Cat video", im)
    cv2.waitKey(1)

    aiEndTime = time.time()
    aiProcessingTime = aiProcessingTime + (aiEndTime - aiBeginTime)
    aiProcessedFrames = aiProcessedFrames + 1
    #eval results
    if (catMatches[0].DistanceToTemplatesMin < 0.3):
        print("cat detected: ", catMatches[0].Name, "dist: ", catMatches[0].DistanceToTemplatesMin)
        haApi.HaPostSensorValue(catMatches[0].Name, "http://hassio.lan:8124")
        if (catMatches[0].Name == "Maugli"):
            pb.SendNotificationWithRateLimiter("Cat detected: " + catMatches[0].Name + " dist:" + f"{catMatches[0].DistanceToTemplatesMin:.2f}")
    else
        if (catMatches[0].DistanceToTemplatesMin < 0.5):
            haApi.HaPostSensorValue("Unconfirmed:"catMatches[0].Name, "http://hassio.lan:8124")
        else
            haApi.HaPostSensorValue("noCat", "http://hassio.lan:8124")


    #outputVideoWrite.write(frame)
#output video stream
#outputVideoWrite = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (1280,720))

#stream
frame_queue = queue.Queue()

def read_frames():
    netstream = cv2.VideoCapture(os.getenv("RSTP_STREAM"))
    readframeCounter = 0
    chunkFrameStart = 0
    chunkTimeStart = time.time()
    while not stop_threads:
        ret, frame = netstream.read()
        if not ret:
            print("Frame is empty")
            time.sleep(0.5)
            continue
        # Put the frame into the queue
        readframeCounter = readframeCounter + 1
        frame_queue.put(frame)
        if (readframeCounter % 100 == 0):
            chunkFrameEnd = readframeCounter
            chunkTimeEnd = time.time()
            print("netstream read: ", (chunkFrameEnd - chunkFrameStart)/(chunkTimeEnd - chunkTimeStart), " FPS: ")
            chunkFrameStart = chunkFrameEnd
            chunkTimeStart = chunkTimeEnd

read_thread = threading.Thread(target=read_frames)
read_thread.start()


def process_framesFromQueue():
    readFrameCounter = 0
    startT = time.time()
    while not stop_threads:
        try:
            #get most recent image from queue and throw away all older images
            while (frame_queue.qsize() > 1):
                frame_queue.get()
            frame = frame_queue.get(True,3)

            readFrameCounter =  readFrameCounter + 1
            
            processFrame(frame, readFrameCounter)
            
            if (readFrameCounter % 100 == 0):   
                ensT = time.time()
                print("read avg FPS: ", readFrameCounter/(ensT - startT))
                print("AI avg FPS: ", aiProcessedFrames/aiProcessingTime)

        except Exception as e:
            print("Error in processFrame")
            print(str(e))  # Print the error message
            traceback.print_exc()  # Print the full traceback
            time.sleep(5)
            
process_thread = threading.Thread(target=process_framesFromQueue)
process_thread.start()

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print('Stopping... wait for threads to finish')
    stop_threads = True
    read_thread.join()
    process_thread.join()
    print('all Threads finished, exiting')
