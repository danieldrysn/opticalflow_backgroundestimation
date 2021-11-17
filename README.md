# Optical Flow Background Estimation

main.py -- provides an optical flow video demo with simplified Python code based upon the following work (original work done in C++): Doyle, D.D., Jennings, A.L., Black, J.T., 'Optical flow background estimation for real-time  pan/tilt camera object tracking, 'Measurement, Vol 48, 2014, Pages 195-207, ISSN 0263-2241, https://doi.org/10.1016/j.measurement.2013.10.025. (https://www.sciencedirect.com/science/article/pii/S0263224113005241)

![Estimation](https://github.com/danieldrysn/opticalflow_backgroundestimation/blob/main/images/helicopter_tracking.jpg)



# Two Image Pan/Tilt Optical Flow Demo

twoimage_pantilt_optflow_demo.py -- brings in two images (left side) and adds good features in blue.  A small pan and tilt motion is made for each image (see right images) and their respective motion is captured with white lines.  Red dots have been removed as outliers.

![Two Image Pan/Tilt Demo](https://github.com/danieldrysn/opticalflow_backgroundestimation/blob/main/images/twoimage_pantilt_optflow_demo.jpg)
