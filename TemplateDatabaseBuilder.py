import argparse
import base64
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


print("Cat template database builder - this program computes tensors from template images and store them in file")

#BASIC SETTINGS
useModelWithReg = True
useColorNormalization = True

baseFolder = "./templates/"
colorImagesFolder = baseFolder + "DayTemplates/"
NightVisionImagesFolder = baseFolder + "NightTemplates/"

#use GPU to speed up
dmlDevice = "cpu"#torch_directml.device()
print("torch version: ", torch.__version__)
print(f'Using device {dmlDevice} for inference')

#load medium size dinov2 model
if useModelWithReg:
    dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
else:
    dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2_vitb14_reg.eval().to(dmlDevice)

#crate array of CatInfo objects
catInfoArray = []
for fileName in os.listdir(colorImagesFolder):
    catInfoArray.append(CatInfo.FromFilePath(colorImagesFolder, fileName))
for fileName in os.listdir(NightVisionImagesFolder):
    catInfoArray.append(CatInfo.FromFilePath(NightVisionImagesFolder, fileName))

#compute tensor for one cat
def computeTensorForCat(catInfo):
    #compute output
    dinoInput = catInfo.GetModelInputTensor(useColorNormalization).to(dmlDevice)
    dinoOutput = dinov2_vitb14_reg(dinoInput)
    outputMainMem = dinoOutput.cpu()
    #get image size
    height, width, channels = catInfo.image.shape
    

    print (f"procssing file {catInfo.imageFilePath}, image dimension:{height}x{width}, output size: ", outputMainMem.size())

    catInfo.outputTensor = outputMainMem[0].detach().numpy()
    #print ("output size: ", catInfo.outputTensor.size())

#compute tensors for all cats
for catInfo in catInfoArray:
    computeTensorForCat(catInfo)
#save cat info array to file
outputFileName = baseFolder + "catsDB" 
if useColorNormalization:
    outputFileName += "_wCN"
if useModelWithReg:
    outputFileName += "_wReg"
outputFileName += ".json"
with open(outputFileName, 'w') as outfile:
    outfile.write("[")
    first = True
    for catInfo in catInfoArray:
        if first:
            first = False
        else:
            outfile.write(",")
        outfile.write(catInfo.ToJson())
    outfile.write("]")

#cv2.imshow('test cat', catInfoArray[1].image)
#cv2.waitKey(5)
#prin all cat info
#for catInfo in catInfoArray:
#    print(catInfo.ToString())

#save cat info array to file



