# Determining Pan and Tilt from Pixel Movement
# Using Least Squares on pixel motion to determin pan and tilt angles
# Daniel D. Doyle
# 2021-11-17

import cv2
import numpy as np
import predictPanTilt as ppt

###########################################################################
#####################    FUNCTION CALLS     ###############################
###########################################################################
    
# Return image with outer area of b pixels as zeros (or specified) for removing features
def make_outerimage_zeros(img,b=100,c=(0,0,0)):
    return cv2.copyMakeBorder(img[b:-b,b:-b],b,b,b,b,cv2.BORDER_CONSTANT,c)

# Set up corner detection and optical flow with its parameters
def init_corners_optflow(corners=100,ql=0.3,mD=10,bS=10,wS=25,mL=3):
    global feature_params, lk_params
    feature_params = dict(maxCorners=corners,qualityLevel=ql,minDistance=mD,blockSize=bS)
    lk_params = dict( winSize = (wS, wS), maxLevel = mL,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,5, 0.03))

# Return good features to track from an image
def grayimg_goodfeaturepts(img):
    img_pad = make_outerimage_zeros(img,20)   # NOTE: Lined features may occur due to zeroing
    gray = cv2.cvtColor(img_pad, cv2.COLOR_BGR2GRAY)
    return gray, cv2.goodFeaturesToTrack(gray, mask = None, **feature_params)      
    
# Return combined image of original (img1) and optical flow (img2)
def show_opticalflow(gray1,gray2,img2,pts0):
    # Calculate optical flow and select good points
    pts1, st, err = cv2.calcOpticalFlowPyrLK(gray1,gray2,pts0,None,**lk_params)
    goodpts1 = pts1[st == 1]
    goodpts0 = pts0[st == 1]
    # Draw the moving points
    for i, (old, new) in enumerate(zip(goodpts0,goodpts1)):
        movecheck = cv2.norm(new-old)
        if movecheck > 4 and movecheck < 30:
            cv2.circle(img2, (int(new[0]),int(new[1])), 5, (208,0,0), -1)
            cv2.line(img2, (int(old[0]),int(old[1])), (int(new[0]),int(new[1])), (255,255,255),2)
        else:
            cv2.circle(img2, (int(old[0]),int(old[1])), 5, (0,0,208), -1)
    return goodpts0.reshape(-1,1,2), goodpts1.reshape(-1,1,2), img2


################################################################################
#########################        MAIN         ##################################
################################################################################
cap = cv2.VideoCapture(0)

init_corners_optflow(corners=100)

ret, img0 = cap.read()
gray0, pts0 = grayimg_goodfeaturepts(img0)

cv2.namedWindow('Pan/Tilt Estimation')

while(True):
    
    ret, img1 = cap.read()
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    
    if pts0.size < 30:
        gray0, pts0 = grayimg_goodfeaturepts(img0)
    img0 = img1 
    pts0, pts1, img1 = show_opticalflow(gray0, gray1, img1, pts0)
    gray0 = gray1   

    # Least Squares approximation of pan and tilt angles
    pan, tilt = ppt.predictPanTiltMovement(480, 640, pts0, pts1)
    pts1 = pts0
    cv2.imshow('Pan/Tilt Estimation', img1)
      
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()

