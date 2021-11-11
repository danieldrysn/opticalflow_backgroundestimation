# Predict Pixel Movement
# Daniel D. Doyle
# 2021-11-10

'''
Predict Pixel Movement umath.sing image rows and columns, focal length, pan, and tilt angles
'''

import math


# Returns predicted point based on size of the image, previous point, focal length, psi, and theta
def predictPixelMovement( rows, cols, previousPoint, foc, psi, theta ):
	xo = previousPoint[0] - cols/2
	yo = previousPoint[1] - rows/2
	xn = foc*((xo - foc*math.tan(psi))/(xo*math.tan(psi)*math.cos(theta)-yo*(math.sin(theta)/math.cos(psi))+foc*math.cos(theta)))
	yn = foc*((xo*math.sin(psi)*math.tan(theta)+yo+foc*math.cos(psi)*math.tan(theta))/(xo*math.sin(psi)-yo*math.tan(theta)+foc*math.cos(psi)))
	return (int(xn + cols/2),int(yn + rows/2))

