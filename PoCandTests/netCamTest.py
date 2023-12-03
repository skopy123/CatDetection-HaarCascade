# USAGE
# python cat_detector.py --image images/cat_01.jpg
from datetime import datetime
# import the necessary packages
import argparse
import cv2
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=false,
# 	help="path to the input image")
ap.add_argument("-c", "--cascade",
	default="haarcascade_frontalcatface.xml",
	help="path to cat detector haar cascade")
args = vars(ap.parse_args())



#camera = cv2.VideoCapture(0)
#camera.set(3,640)
#camera.set(4,480)
netstream = cv2.VideoCapture('rtsp://maugli:vetrelec1@192.168.129.198:554/stream1')

# load the cat detector Haar cascade, then detect cat faces in the input image
detector = cv2.CascadeClassifier(args["cascade"])

directory = "./cats"

# load the input image and convert it to grayscale
#image = cv2.imread(args["image"])
lastImageSaveTime = datetime.now()
while (1):
	ret, image = netstream.read()
	now = datetime.now()
	if ret == False:
		print("Frame is empty")
		break;
	else:
		cv2.GaussianBlur(image, (7, 7), 0) 

		cv2.imshow('VIDEO', image)
		cv2.waitKey(1)

