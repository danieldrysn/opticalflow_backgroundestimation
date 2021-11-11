# Optical Flow Background Estimation (simplified Python version)
# Add your own intrinsics and extrinsics for pan/tilt/zoom motiongit
# Daniel D. Doyle
# 2021-11-10 

'''
This simplified Python code is based upon the following work (original work done in C++):
Doyle, D.D., Jennings, A.L., Black, J.T., 'Optical flow background estimation for real-time 
pan/tilt camera object tracking, 'Measurement, Vol 48, 2014, Pages 195-207, ISSN 0263-2241,
https://doi.org/10.1016/j.measurement.2013.10.025.
(https://www.sciencedirect.com/science/article/pii/S0263224113005241)
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
feature_params = dict(maxCorners=500,qualityLevel=0.3,minDistance=7,blockSize=7)
lk_params = dict( winSize = (15, 15), maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10, 0.03))
  
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
        if movecheck > 4:
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
    if p0.size < 50:
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
  
cv2.destroyAllWindows()
cap.release()