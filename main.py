# Optical Flow Background Estimation (simplified Python version)
# Add your own intrinsics and extrinsics for pan/tilt/zoom motion
# Daniel D. Doyle
# 2021-11-10 

'''
This simplified Python code is based upon the following work (original work done in C++)
'''

import numpy as np
import cv2
import predictPixelMovement as pm 
import cameraParameters as cam

cap = cv2.VideoCapture(0)

# Get camera intrinsics, extrinsics, and video parameters
foc,cx,cy,skew = cam.getCameraIntrinsics(cap)
rows,cols,fps = cam.getVideoParameters(cap)
scale,psi,theta = cam.getCameraExtrinsics()

# Parameters for corner detection and Lucas Kanade Optical Flow
feature_params = dict(maxCorners=200,qualityLevel=0.3,minDistance=10,blockSize=10)
lk_params = dict( winSize = (25, 25), maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,5, 0.03))
  
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Find features on the frame
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
      
while(True):
    
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  
    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,p0,None,**lk_params)
    
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
  
    # Draw the moving points
    for i, (new, old) in enumerate(zip(good_new,good_old)):
        predicted = pm.predictPixelMovement(rows, cols, old, foc, psi, theta)
        movecheck = cv2.norm(predicted-new)
        if movecheck > 4 and movecheck < 50:
            frame = cv2.circle(frame, predicted, 5, (208,0,0), -1)
            frame = cv2.line(frame, tuple(old), tuple(new), (255,255,255),2)
        else:
            frame = cv2.circle(frame, predicted, 5, (0,0,208), -1)
          
    cv2.imshow('frame', frame)
      
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Updating Previous frame and points 
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    if p0.size < 10:
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
  
cv2.destroyAllWindows()
cap.release()