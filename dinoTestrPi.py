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

#resnet50.eval().to(device)

# load the input image and convert it to grayscale
image1 = cv2.imread("./images/cat_01.jpg")
#cv2.imshow('cat image', image)
#cv2.waitKey(0)
image2 = cv2.imread("./images/cat_02.jpg")

dim = (140, 140)
# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor()
])




image1 = cv2.resize(image1, dim, interpolation = cv2.INTER_AREA)
preprocess = transforms.Compose([
    # convert the frame to a CHW torch tensor for training
    transforms.ToTensor(),
    # normalize the colors to the range that mobilenet_v2/3 expect
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image1)
# The model can handle multiple images simultaneously so we need to add an
# empty dimension for the batch.
# [3, 224, 224] -> [1, 3, 224, 224]
input_batch = input_tensor.unsqueeze(0)



#dim = (224, 168)


"""

images = [
    transform(cv2.resize(cv2.imread("./images/maugli1.jpg"),dim, interpolation = cv2.INTER_LINEAR)).unsqueeze(0),
    transform(cv2.resize(cv2.imread("./images/maugli1.jpg"),dim, interpolation = cv2.INTER_AREA)).unsqueeze(0),
    transform(cv2.resize(cv2.imread("./images/maugli2.jpg"),dim, interpolation = cv2.INTER_AREA)).unsqueeze(0),
    transform(cv2.resize(cv2.imread("./images/maugli3.jpg"),dim, interpolation = cv2.INTER_AREA)).unsqueeze(0),
  
    transform(cv2.resize(cv2.imread("./images/amalka.jpg"),dim, interpolation = cv2.INTER_AREA)).unsqueeze(0),
    transform(cv2.resize(cv2.imread("./images/gita1.jpg"),dim, interpolation = cv2.INTER_AREA)).unsqueeze(0),
    transform(cv2.resize(cv2.imread("./images/gita2.jpg"),dim, interpolation = cv2.INTER_AREA)).unsqueeze(0),
    #transform(cv2.resize(cv2.imread("./images/kocour1.jpg"),dim, interpolation = cv2.INTER_AREA)).unsqueeze(0),
  
    #transform(cv2.imread("./images/amalka.jpg")).unsqueeze(0),
]
#prepared_images = [utils.prepare_input_from_uri(uri) for uri in uris]
"""

imageNames = [
    "m1",
    "m2",
    "m3",
    "g1",
    "g2",
    #"g3",
]


#images = []
#for imageName in imageNames:
#    images.append(transform(cv2.imread("./images/newTemplates/"+imageName+".png")).unsqueeze(0))

startTime = time.time()

#input_batch = torch.cat(
#   #prepared_images 
#  images
#).to(device)

dinoOutput = dinov2_vits14_reg(input_batch).cpu()
print ("output size: ", dinoOutput.size())
#cretate array of numpy tensors
knownCatsTensors = []
for i in range(0,imageNames.__len__()):
    knownCatsTensors.append(dinoOutput[i].detach().numpy())
endTime = time.time()
print ("CPU processing time (1 images): ", endTime - startTime)

#print ("dinoOutput: ", transformedOutput)
exit()

#define print function, input is two indexes, it read names from imageNames array and compute cosine distance between coresponding tensors
def printCosineDistance(index1, index2):
    print ("distance " + imageNames[index1] + "-" + imageNames[index2] + ": " , cosine(knownCatsTensors[index1], knownCatsTensors[index2]))

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
        tensor = cosine(unknownTensor, knownCatsTensors[i])
        endTime2 = time.time()
        print ("distance " + fileName + "-" + imageNames[i] + ": " , tensor,  " time: ", endTime2 - startTime2)

print ("compare to unknown images")
compareUnknownImageToDB("./images/maugli1.jpg")
compareUnknownImageToDB("./images/maugli2.jpg")
compareUnknownImageToDB("./images/maugli3.jpg")
compareUnknownImageToDB("./images/gita1.jpg")
compareUnknownImageToDB("./images/gita2.jpg")

exit()
print ("ditance 0-7: ", cosine(knownCatsTensors[0], knownCatsTensors[7]))

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