
import argparse
import base64
import json
import cv2
from torchvision import models,transforms
import numpy as np

#cat names to abreaviation dictionary
nameDictionary = {
    "a" : "Amalka",
    "g" : "Gita",
    "k" : "Kocour",
    "m" : "Maugli",
}

# define class for storing image properties and tensor
class CatInfo:
    @staticmethod
    def FromFilePath(folder, fileName):
        ci = CatInfo()
        ci.imageFilePath = folder + fileName
        substring = "night"
        ci.nightVision = substring in folder.lower()

        # fileName is like m1.png, g2.png, etc - we need to extract the cat name
        #from filename remove extension behinf dot
        fn = fileName.split(".")[0]
        #find first digit in fn
        digitIndex = -1
        for i in range(len(fn)):
            if fn[i].isdigit():
                digitIndex = i
                break
        #extract cat name
        ci.catName = nameDictionary[fn[:digitIndex]]
        #extract cat number
        ci.catImageNumber = int(fn[digitIndex:])
        #load image
        ci.image = cv2.imread(ci.imageFilePath)
        #resize image, result image resulution must me multiplier of 14. londer side must be 336px or less, keep aspect ratio as close as possible
        #find longer side
        longerSide = max(ci.image.shape[0], ci.image.shape[1])
        #find multiplier
        scaleFactor = 1
        if longerSide > 280:
            scaleFactor = 280 / longerSide
        #resize image, new width must be devidable by 14
        newWidth = int(ci.image.shape[1] * scaleFactor / 14) * 14
        newHeight = int(ci.image.shape[0] * scaleFactor / 14) * 14
        newDim = (newWidth, newHeight)
        ci.image = cv2.resize(ci.image, newDim, interpolation = cv2.INTER_AREA)
        ci.outputTensor = None
        return ci
    
    #parameterless ctor for import
    def __init__(self):
        self.imageFilePath = ""
        self.nightVision = False
        self.catName = ""
        self.catImageNumber = 0
        self.image = None
        self.outputTensor = None
    
    preprocess_withNormalize = transforms.Compose([
    	# convert the frame to a CHW torch tensor for training
    	transforms.ToTensor(),
    	# normalize the colors to the range that mobilenet_v2/3 expect
	    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    preprocess = transforms.Compose([
        transforms.ToTensor()
    ])


    def GetModelInputTensor(self, useColorNormalization = True):
        if useColorNormalization:
            return self.preprocess_withNormalize(self.image).unsqueeze(0)
        else:
            return self.preprocess(self.image).unsqueeze(0)

    def ToString(self):
        return f"CatImageInfo: {self.catName}{self.catImageNumber} nv: {self.nightVision}"
    
    def ShortDescription(self):
        nv = ""
        if self.nightVision:
            nv = "NV"
        return f"{self.catName}{self.catImageNumber}{nv}"
    
    def ToJson(self):
        #serialize to json, encode tensor to base64 string
        outputTensorList = self.outputTensor.tolist()
        return json.dumps({"catName": self.catName, "catImageNumber": self.catImageNumber, "nightVision": self.nightVision, "outputTensor": outputTensorList})
    
    #ctor with deserialization from json
    @staticmethod
    def FromJson(jsonObject):
        #deserialize from json, decode tensor from base64
        #jsonObject = json.loads(jsonString)
        catInfo = CatInfo()
        catInfo.catName = jsonObject["catName"]
        catInfo.catImageNumber = jsonObject["catImageNumber"]
        catInfo.nightVision = jsonObject["nightVision"]
        #catInfo.outputTensor = np.frombuffer(base64.b64decode(jsonObject["outputTensor"]))
        catInfo.outputTensor = np.array(jsonObject["outputTensor"])

        return catInfo
