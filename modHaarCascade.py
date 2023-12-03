#modHaarCascade for detection cat faces in image
import cv2
import numpy as np
import time
#haar detector
detector = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")

firFilterLength = 20
catCounterFIRfilter = np.zeros(firFilterLength)

lastImageCatCount = -1
lastImageTime = time.time()
lastImageCatsBoundingBoxes = []
catPresetnFiltred = False


#input image must be grayscale, retuns array of bounding boxes
def DetectCatFaces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.3,minNeighbors=10, minSize=(75, 75))
    return rects

#image must be grayscale
def PutImageIntoProcessPipeline(image):
    global lastImageCatCount
    global lastImageTime
    global lastImageCatsBoundingBoxes
    global catCounterFIRfilter
    global catPresetnFiltred
    lastImageCatsBoundingBoxes = DetectCatFaces(image)
    lastImageCatCount = lastImageCatsBoundingBoxes.__len__()
    lastImageTime = time.time()
    #FIR filter
    #TODO reset fir filter if time difference is too big

    catCounterFIRfilter.pop(0)
    if (lastImageCatCount > 0):
        catCounterFIRfilter.append(1)
    else:
        catCounterFIRfilter.append(0)
    filterOutValue = sum(catCounterFIRfilter)/len(catCounterFIRfilter)

    catPresetnFiltred = ((lastImageCatCount > 0) or (filterOutValue > 0.05))

def AddBoundingBoxesIntoImage(image):
    for (i, (x, y, w, h)) in enumerate(lastImageCatsBoundingBoxes):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
