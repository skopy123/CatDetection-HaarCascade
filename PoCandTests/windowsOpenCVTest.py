# USAGE
# python cat_detector.py --image images/cat_01.jpg

# import the necessary packages
import argparse
import cv2
import os
import torch
import torch_directml
import numpy as np
from torchvision import models,transforms

import torch.nn as nn

from torchvision.models import resnet50
from torch.nn.functional import normalize
from tqdm import tqdm
from scipy.spatial.distance import cosine


print("torch and dml test")
dml = torch_directml.device()
tensor1 = torch.tensor([1]).to(dml) # Note that dml is a variable, not a string!
tensor2 = torch.tensor([2]).to(dml)
dml_algebra = tensor1 + tensor2
dml_algebra.item()
print(dml_algebra.item())

print("torch version: ", torch.__version__)
#exit()

device = dml
print(f'Using {device} for inference')

#torch.backends.quantized.engine = 'qnnpack'
#net = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
# jit model to take it from ~20fps to ~30fps
#net = torch.jit.script(net)

resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

model_dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
model_dino.eval().to(device)

#resnet50.eval().to(device)

# load the input image and convert it to grayscale
image1 = cv2.imread("./images/cat_01.jpg")
#cv2.imshow('cat image', image)
#cv2.waitKey(0)
image2 = cv2.imread("./images/cat_02.jpg")

# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor()
])


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

uris = [
    'http://images.cocodataset.org/test-stuff2017/000000024309.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000028117.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000006149.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000004954.jpg',
]

images = [
    #transform(cv2.imread("./images/cat_01.jpg")).unsqueeze(0),
    #transform(cv2.imread("./images/cat_02.jpg")).unsqueeze(0),
    #transform(cv2.imread("./images/cat_03.jpg")).unsqueeze(0),
    #transform(cv2.imread("./images/cat_04.jpg")).unsqueeze(0)
    transform(cv2.imread("./images/maugli1.jpg")).unsqueeze(0),
    transform(cv2.imread("./images/maugli2.jpg")).unsqueeze(0),
    transform(cv2.imread("./images/maugli3.jpg")).unsqueeze(0),
    transform(cv2.imread("./images/kocour1.jpg")).unsqueeze(0),
    transform(cv2.imread("./images/kocour2.jpg")).unsqueeze(0),
    transform(cv2.imread("./images/amalka.jpg")).unsqueeze(0),
    transform(cv2.imread("./images/gita1.jpg")).unsqueeze(0),
    transform(cv2.imread("./images/gita2.jpg")).unsqueeze(0),

]
prepared_images = [utils.prepare_input_from_uri(uri) for uri in uris]

input_batch = torch.cat(
   #prepared_images 
  images
).to(device)

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