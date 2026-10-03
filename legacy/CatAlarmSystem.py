
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
import modHaHttpSensorPush as haApi

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
    else:
        if (catMatches[0].DistanceToTemplatesMin < 0.5):
            haApi.HaPostSensorValue("Unconfirmed:" + catMatches[0].Name, "http://hassio.lan:8124")
        else:
            haApi.HaPostSensorValue("noCat", "http://hassio.lan:8124")


    #outputVideoWrite.write(frame)
#output video stream
#outputVideoWrite = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (1280,720))

#stream
frame_queue = queue.Queue()

def read_frames():
    stream_url = os.getenv("RSTP_STREAM")
    netstream = cv2.VideoCapture(stream_url)
    readframeCounter = 0
    chunkFrameStart = 0
    chunkTimeStart = time.time()

    # Reconnection parameters
    reconnect_attempts = 0
    max_reconnect_attempts = 10
    reconnect_delay = 2  # Initial delay in seconds

    while not stop_threads:
        ret, frame = netstream.read()
        
        if not ret:
            print("Frame is empty, attempting to reconnect...")
            reconnect_attempts += 1
            if reconnect_attempts > max_reconnect_attempts:
                print(f"Max reconnect attempts reached. Reinitializing VideoCapture.")
                netstream.release()  # Release the current capture object
                time.sleep(reconnect_delay)  # Wait before attempting to reconnect
                netstream = cv2.VideoCapture(stream_url)
                reconnect_attempts = 0
                reconnect_delay = min(reconnect_delay * 2, 60)  # Exponential backoff, capped at 60 seconds
            else:
                time.sleep(0.5)  # Short sleep before the next attempt
            continue

        # Reset reconnection attempts if a frame is successfully read
        reconnect_attempts = 0
        reconnect_delay = 2  # Reset delay to the initial value

        # Put the frame into the queue
        readframeCounter += 1
        frame_queue.put(frame)
        
        # Logging frame processing statistics
        if readframeCounter % 100 == 0:
            chunkFrameEnd = readframeCounter
            chunkTimeEnd = time.time()
            fps = (chunkFrameEnd - chunkFrameStart) / (chunkTimeEnd - chunkTimeStart)
            print(f"netstream read: {fps:.2f} FPS")
            chunkFrameStart = chunkFrameEnd
            chunkTimeStart = chunkTimeEnd

    # Release the VideoCapture when stopping
    netstream.release()

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
