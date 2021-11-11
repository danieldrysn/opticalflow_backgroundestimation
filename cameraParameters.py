# Camera and Video Parameters
# Daniel D. Doyle
# 2021-11-10

import cv2

def getCameraIntrinsics(cap):
    focal_length = 1
    center_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
    center_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)
    axis_skew = 0
    return focal_length, center_x, center_y, axis_skew

def getVideoParameters(cap):
    if cap.isOpened(): # get height (rows), width (columns), and frames per second (fps)
        rows = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  
        cols  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f'Image rows={rows}, cols={cols}, fps={fps}')  
        return rows, cols, fps

def getCameraExtrinsics(scale = 1):
    psi = 0
    theta = 0
    return scale, psi, theta