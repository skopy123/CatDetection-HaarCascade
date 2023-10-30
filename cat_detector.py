# USAGE
# python cat_detector.py --image images/cat_01.jpg

# import the necessary packages
import argparse
import cv2
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=false,
# 	help="path to the input image")
ap.add_argument("-c", "--cascade",
	default="haarcascade_frontalcatface.xml",
	help="path to cat detector haar cascade")
args = vars(ap.parse_args())



#camera = cv2.VideoCapture(0)
netstream = cv2.VideoCapture('rtsp://192.168.128.113:554/live')

# load the cat detector Haar cascade, then detect cat faces in the input image
detector = cv2.CascadeClassifier(args["cascade"])

# load the input image and convert it to grayscale
#image = cv2.imread(args["image"])
while (1):
	ret, image = netstream.read()
	if ret == False:
		print("Frame is empty")
		break;
	else:
		cv2.imshow('VIDEO', image)
		cv2.waitKey(1)

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		rects = detector.detectMultiScale(gray, scaleFactor=1.3,minNeighbors=10, minSize=(75, 75))
		catCount = 0
# loop over the cat faces and draw a rectangle surrounding each
		for (i, (x, y, w, h)) in enumerate(rects):
			cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
			cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
			catCount = catCount + 1

#write the image
#cv2.imwrite('result.jpg', image)
# show the detected cat faces
		if (catCount > 0):
			cv2.imshow("Cat Faces", image)
			cv2.waitKey(1)
#cv2.destroyAllWindows()