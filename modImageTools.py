#modImageTools
import cv2
import numpy as np

#returns true if image is BW(night vision)
def DetectNigthVision(image):
    ### splitting b,g,r channels
    b,g,r=cv2.split(image)

    ### getting differences between (b,g), (r,g), (b,r) channel pixels
    r_g=np.count_nonzero(abs(r-g))
    r_b=np.count_nonzero(abs(r-b))
    g_b=np.count_nonzero(abs(g-b))

    ### sum of differences
    diff_sum=float(r_g+r_b+g_b)

    ### finding ratio of diff_sum with respect to size of image
    ratio=diff_sum/image.size

    print("ratio: ", ratio)
    if ratio>0.1:
        #print("image is color")
        return False
    else:
        #print("image is greyscale")
        return True

def ConvertToGrayScale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def ConvertToGrayScaleIfNecessary(image):
    if DetectNigthVision(image):
        return ConvertToGrayScale(image)
    else:
        return image