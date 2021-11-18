# Determining Pan and Tilt from Pixel Movement
# Two image analysis of pan and tilt on a USAF Japan Man
# Daniel D. Doyle
# 2021-11-15

import cv2
import numpy as np
import predictPixelMovement as pm 
import predictPanTilt as ppt 

###########################################################################
#####################    FUNCTION CALLS     ###############################
###########################################################################

# Return concatenated image set using [[1,..,n],[n+1,..,m]]
def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])
    
# Return image with outer area of b pixels as zeros (or specified) for removing features
def make_outerimage_zeros(img,b=100,c=(0,0,0)):
    return cv2.copyMakeBorder(img[b:-b,b:-b],b,b,b,b,cv2.BORDER_CONSTANT,c)

# Return image with features at pts with specified radius and color
def display_features(img,pts,radius=5,color=(255,255,255),text=''):
    for i,pt in enumerate(pts):
        cv2.circle(img,(int(pt[0]),int(pt[1])),radius,color,-1)
    cv2.putText(img, text, (10,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, color, 2)

# Return image with lines between pts1 and pts2 with specified color
def display_lines(img,pts1,pts2,color=(255,255,255)):
    for pt1,pt2 in enumerate(zip(pts1,pts2)):
        cv2.line(img, (int(pt[0]),int(pt[1]), 5, color, thickness))

# Provide title, images as [[1,..,n],[n+1,..,m]], and final scale
def display_images(title,images,scale=1):
    cv2.namedWindow(title)
    img = concat_tile(images)
    img = cv2.resize(img,(round(img.shape[1]*scale),round(img.shape[0]*scale)),interpolation=cv2.INTER_AREA)
    cv2.imshow(title,img)
    cv2.waitKey(0)
    return img

# Set up corner detection and optical flow with its parameters
def init_corners_optflow(corners=100,ql=0.3,mD=10,bS=10,wS=25,mL=3):
    global feature_params, lk_params
    feature_params = dict(maxCorners=corners,qualityLevel=ql,minDistance=mD,blockSize=bS)
    lk_params = dict( winSize = (wS, wS), maxLevel = mL,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,5, 0.03))

# Return combined image of original (img1) and optical flow (img2)
def show_opticalflow(img1,img2,text=''):
    img_pad = make_outerimage_zeros(img1)
    gray1 = cv2.cvtColor(img_pad, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    # Find good feature to track and return good points
    pts0 = cv2.goodFeaturesToTrack(gray1, mask = None, **feature_params)      
    # Calculate optical flow and select good points
    pts1, st, err = cv2.calcOpticalFlowPyrLK(gray1,gray2,pts0,None,**lk_params)
    good_pts1 = pts1[st == 1]
    good_pts0 = pts0[st == 1]
    move_pts0, move_pts1 = [],[]
    # Draw old points on Gray1 and new points on Gray 2
    display_features(img1, good_pts0,5,(208,0,0),'Corners')
    # Draw the moving points
    for i, (new, old) in enumerate(zip(good_pts1,good_pts0)):
        predicted = pm.predictPixelMovement(480, 640, old, 1, 0, 0)
        movecheck = cv2.norm(predicted-new)
        if movecheck > 0 and movecheck < 50:
            cv2.circle(img2, predicted, 5, (208,0,0), -1)
            cv2.line(img2, (int(old[0]),int(old[1])), (int(new[0]),int(new[1])), (255,255,255),2)
            move_pts0.append(old)
            move_pts1.append(new)
        else:
            cv2.circle(img2, predicted, 5, (0,0,208), -1)
    move0 = np.array(move_pts0)
    move1 = np.array(move_pts1)
    pan, tilt = ppt.predictPanTiltMovement(480, 640,move0.reshape((-1,1,2)),move1.reshape(-1,1,2))
    text2 = f'Pan: {pan:2.2f}, Tilt: {tilt:2.2f}'
    cv2.putText(img2, text, (10,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(208,0,0),2)
    cv2.putText(img2, text2, (700,700), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255,255,255),2)
    return concat_tile([[img1,img2]])


################################################################################
#########################        MAIN         ##################################
################################################################################
# Load the images
img1 = cv2.imread('images/USAF-JapanMan-Pan1.jpg')
img2 = cv2.imread('images/USAF-JapanMan-Pan2.jpg')
img3 = cv2.imread('images/USAF-JapanMan-Tilt1.jpg')
img4 = cv2.imread('images/USAF-JapanMan-Tilt2.jpg')

init_corners_optflow(corners=100)
imgpan = show_opticalflow(img1, img2,'Pan')
imgtilt = show_opticalflow(img3, img4,'Tilt')
imgfinal = display_images('USAF-JapanMan',[[imgpan],[imgtilt]],0.3)
cv2.imwrite('twoimage_pantilt_optflow_demo.jpg',imgfinal)
cv2.destroyAllWindows()