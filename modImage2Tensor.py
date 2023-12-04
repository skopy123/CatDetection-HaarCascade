#get image and compute tensor by dinov2 model
import os
import torch

from torchvision import models,transforms
import torch.nn as nn
from torch.nn.functional import normalize
import modSettings as settings
import cv2

#load model
dmlDevice = "cpu"
if (settings.useDirectML):
    import torch_directml
    dmlDevice = torch_directml.device()


#load medium size dinov2 model
if settings.useModelWithReg:
    dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
else:
    dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2_vitb14_reg.eval().to(dmlDevice)

def GetModelInputTensor(image, useColorNormalization = True):
        preprocess = transforms.Compose([
            transforms.ToTensor()
        ])
        if useColorNormalization:
            preprocess = transforms.Compose([
    	    # convert the frame to a CHW torch tensor for training
    	    transforms.ToTensor(),
    	    # normalize the colors to the range that mobilenet_v2/3 expect
	        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        return preprocess(image).unsqueeze(0)


#compute tensor for one image
def ComputeTensorForImage(image: cv2.typing.MatLike):
    #compute output
    dinoInput = GetModelInputTensor(image, settings.useColorNormalization).to(dmlDevice)
    dinoOutput = dinov2_vitb14_reg(dinoInput)
    outputMainMem = dinoOutput.cpu()
    #get image size
    height, width, channels = image.shape
    print (f"procssing image, dimension:{height}x{width}, output tensor size: ", outputMainMem.size())
    return outputMainMem[0].detach().numpy()

def ResizeImageToModelInput(image: cv2.typing.MatLike, maxHeight = 280) -> cv2.typing.MatLike:
    #if dimension of image is not devidable by 14, resize it
    oldHeight = image.shape[0]
    scale = 1
    if(oldHeight>maxHeight):
        scale = maxHeight/oldHeight
    newWidth = int(image.shape[1]*scale/14)*14        
    newHeight = int(image.shape[0]*scale/14)*14
    newDim = (newWidth, newHeight)
    scaledim = cv2.resize(image, newDim, interpolation = cv2.INTER_AREA)
    return scaledim