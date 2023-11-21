# USAGE
# python cat_detector.py --image images/cat_01.jpg

# import the necessary packages
import argparse
import cv2
import os
import torch
#import torch_directml
import time
import numpy as np
from torchvision import models,transforms

import torch.nn as nn

from torchvision.models import resnet50
from torch.nn.functional import normalize
from tqdm import tqdm
from scipy.spatial.distance import cosine



print("torch version: ", torch.__version__)
#exit()

device = "cpu"#dml
print(f'Using {device} for inference')

print("torch and dml test")
dml = device
#tensor1 = torch.tensor([1]).to(dml) # Note that dml is a variable, not a string!
#tensor2 = torch.tensor([2]).to(dml)
#dml_algebra = tensor1 + tensor2
#dml_algebra.item()
#print(dml_algebra.item())

#exit()

#torch.backends.quantized.engine = 'qnnpack'
#net = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
# jit model to take it from ~20fps to ~30fps
#net = torch.jit.script(net)

#resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
#utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

#model_dino = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8', pretrained=True)
#model_dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
#dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
#dinov2_vitb14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')
#dinov2_vitb14_reg.eval().to(device)

dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
dinov2_vits14_reg.eval().to(device)


preprocess = transforms.Compose([
    # convert the frame to a CHW torch tensor for training
    transforms.ToTensor(),
    # normalize the colors to the range that mobilenet_v2/3 expect
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def loadImage(fileName, width, height):
    dim = (width, height)
    image1 = cv2.imread(fileName)
    image1 = cv2.resize(image1, dim, interpolation = cv2.INTER_LINEAR)
    input_tensor = preprocess(image1)
    return input_tensor.unsqueeze(0)



imageNames = []
resultTensors = []

def ComputeTensorForImage(fileName):
    imageNames.append(fileName)
    startTime = time.time()
    modelInput = loadImage(fileName, 224, 224)
    #compute output
    dinoOutput = dinov2_vits14_reg(modelInput)#.cpu()
    print ("output size: ", dinoOutput.size())
    #print ("output: ", dinoOutput)
    resultTensors.append(dinoOutput[0].detach().numpy())
    endTime = time.time()
    print ("CPU processing time (1 images): ", endTime - startTime)


#ComputeTensorForImage("./images/cat_01.jpg")
#ComputeTensorForImage("./images/cat_03.jpg")

ComputeTensorForImage("./testImg/maugli1.jpg")
ComputeTensorForImage("./testImg/maugli2.jpg")

#print ("dinoOutput: ", transformedOutput)

#define print function, input is two indexes, it read names from imageNames array and compute cosine distance between coresponding tensors
def printCosineDistance(index1, index2):
    print ("distance " + imageNames[index1] + "-" + imageNames[index2] + ": " , cosine(resultTensors[index1], resultTensors[index2]))

printCosineDistance(0, 1)
exit()

#result for maugli
def computeDeltas(referenceIndex):
    for i in range(0,imageNames.__len__()):
        if (i != referenceIndex):
            printCosineDistance(referenceIndex, i)

print("debug info about distances between images in database")
computeDeltas(0);
computeDeltas(3);
print("end debug info")
def GetTensorForUnknownImage(fileName):
    #load image
    im = cv2.imread(fileName)
    #if dimension of image is not devidable by 14, resize it
    if (im.shape[0] % 14 != 0 or im.shape[1] % 14 != 0):
        newWidth = im.shape[1] - (im.shape[1] % 14)
        newHeight = im.shape[0] - (im.shape[0] % 14)
        newDim = (newWidth, newHeight)
        newDim = (224, 168)
        newDim = (168, 126)
        im = cv2.resize(im, newDim, interpolation = cv2.INTER_AREA)
    #transform image to tensor and copy to GPU
    imArray = [
        transform(im).unsqueeze(0)
    ]
    modelInputBatch = torch.cat(imArray).to(device)
    #compute output
    modelOutput = dinov2_vits14_reg(modelInputBatch).cpu()
    #return tensor
    return modelOutput[0].detach().numpy()


def compareUnknownImageToDB(fileName):
    startTime2 = time.time()
    unknownTensor = GetTensorForUnknownImage(fileName)
    for i in range(0,imageNames.__len__()):
        tensor = cosine(unknownTensor, resultTensors[i])
        endTime2 = time.time()
        print ("distance " + fileName + "-" + imageNames[i] + ": " , tensor,  " time: ", endTime2 - startTime2)

print ("compare to unknown images")
compareUnknownImageToDB("./images/maugli1.jpg")
compareUnknownImageToDB("./images/maugli2.jpg")
compareUnknownImageToDB("./images/maugli3.jpg")
compareUnknownImageToDB("./images/gita1.jpg")
compareUnknownImageToDB("./images/gita2.jpg")

exit()
print ("ditance 0-7: ", cosine(resultTensors[0], resultTensors[7]))

exit()
   # run model
#output = net(input_batch)
print("input_batch: ", input_batch.size())
#top = list(enumerate(output[0].softmax(dim=0)))
#top.sort(key=lambda x: x[1], reverse=True)
#for idx, val in top[:10]:
#    print(f"{val.item()*100:.2f}% {idx}")

with torch.no_grad():
    output = torch.nn.functional.softmax(resnet50(input_batch), dim=1)
    
results = utils.pick_n_best(predictions=output, n=5)

#for  result in zip( results):
    #img = Image.open(requests.get(uri, stream=True).raw)
    #img.thumbnail((256,256), Image.ANTIALIAS)
    #plt.imshow(img)
    #plt.show()
    #print(result)