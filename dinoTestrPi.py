# USAGE
# python cat_detector.py --image images/cat_01.jpg

# import the necessary packages
import argparse
import cv2
import os
import torch
#import torch_directml
import time
from datetime import datetime
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

#dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
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
ComputeTensorForImage("./testImg/gita1.jpg")
ComputeTensorForImage("./testImg/gita2.jpg")


#print ("dinoOutput: ", transformedOutput)

#define print function, input is two indexes, it read names from imageNames array and compute cosine distance between coresponding tensors
def printCosineDistance(index1, index2):
	print ("distance " + imageNames[index1] + "-" + imageNames[index2] + ": " , cosine(resultTensors[index1], resultTensors[index2]))

printCosineDistance(0, 1)
printCosineDistance(2, 3)

printCosineDistance(0, 2)
printCosineDistance(0, 3)

camera = cv2.VideoCapture(0)
camera.set(3,640)
camera.set(4,480)

detector = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")

directory = "./cats"

# load the input image and convert it to grayscale
#image = cv2.imread(args["image"])
lastImageSaveTime = datetime.now()
while (1):
	ret, image = camera.read()
	now = datetime.now()
	if ret == False:
		print("Frame is empty")
		break;
	else:
	#	cv2.imshow('VIDEO', image)
	#	cv2.waitKey(1)

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		rects = detector.detectMultiScale(gray, scaleFactor=1.3,minNeighbors=10, minSize=(75, 75))
		catCount = 0
# loop over the cat faces and draw a rectangle surrounding each
		for (i, (x, y, w, h)) in enumerate(rects):
			#enlarge roundBox by 50%
			scale = 1.5
			x = x - (w*scale - w)/2
			y = y - (h*scale - h)/2
			if (x < 0):
				x = 0
			if (y < 0):
				y = 0
			w = w*scale
			h = h*scale
			# round w and h to be devidaible by 14, if not round up
			w = (w//14)*14 + (w%14 > 0)*14
			h = (h//14)*14 + (h%14 > 0)*14

			startTime = time.time()
			cropImage = image[int(y):int(y+h), int(x):int(x+w)]
			input_tensor = preprocess(cropImage).unsqueeze(0)
			dinoOutput2 = dinov2_vits14_reg(input_tensor)
			unknownCatTensor = dinoOutput[0].detach().numpy()
			cosM1 = cosine(unknownCatTensor, resultTensors[0])
			cosM2 = cosine(unknownCatTensor, resultTensors[1])
			endTime = time.time()
			
			print ("CPU processing time (1 images): ", endTime - startTime)

			message = "Cat #{}, cosDst-m1:{:.3f}, cosDst-m2:{:.3f}".format(i + 1, cosM1, cosM2)
			cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
			cv2.putText(image, message, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
			catCount = catCount + 1

#write the image
#cv2.imwrite('result.jpg', image)
# show the detected cat faces
	#	if (catCount > 0):
		cv2.putText(image, now.strftime("%Y-%m-%d %H:%M:%S"), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
			
		#cv2.imshow("Cat Faces", image)
		#cv2.waitKey(1)

		saveImageEnable = ((now - lastImageSaveTime).total_seconds() > 2)
		if saveImageEnable and (catCount > 0):
			lastImageSaveTime = now
			filename = directory + "/" + now.strftime("%Y-%m-%d-%H-%M-%S")+".jpg"
			cv2.imwrite(filename, image) 



#cv2.destroyAllWindows()
