# Predict Pan/Tilt Movement using Simplified Linear Approach
# Daniel D. Doyle
# 2021-11-12

'''	
Predict pan and tilt movement using pixel motion
'''

import numpy as np
import math
import predictPixelMovement as pm

# Return predicted pan/tilt based on size of the image, previous and new points and focal length
def predictPanTiltMovement( rows, cols, prevPoints, newPoints):
	eps = 1
	f = 1
	U, pxpy = np.zeros([prevPoints.size,2]), np.zeros([prevPoints.size,2,2])
	Ug = np.zeros([2,1])
	
	diffPoints = newPoints - prevPoints
	
	avgx = np.average(diffPoints[:,0,0])
	avgy = np.average(diffPoints[:,0,1])

	print(f'Pan = {float(avgx):2.4f}, Tilt = {float(avgy):2.4f}')
	return avgx,avgy