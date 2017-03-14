import cv2
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

import math


def my_Normalize(img):

	if img is not None:
		rows, columns, channels = img.shape

		if channels != 1:
			grayScaleImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		else:
			grayScaleImg = img

		normalizedImg = cv2.normalize(grayScaleImg.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
		return normalizedImg

	else:
		print('No image input or wrong input.')
		return

def my_DerivativesOfGaussian(img, sigma):
	sobelx, sobely = sobelKernel()
	myGaussianFilter = GaussianFilter(sigma)

	Ix = cv2.filter2D(img, -1, sobelx)
	Iy = cv2.filter2D(img, -1, sobely)
	IxGauss = cv2.filter2D(Ix, -1, myGaussianFilter)
	IyGauss = cv2.filter2D(Iy, -1, myGaussianFilter)

	IxNormalized = cv2.normalize(IxGauss.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	IyNormalized = cv2.normalize(IyGauss.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
#cv2.cvtColor(imgYCC, cv2.COLOR_YCR_CB2BGR)
	
	cv2.imshow('IxNormalized', IxNormalized)
	cv2.imshow('IyNormalized', IyNormalized)

	cv2.waitKey(0)
	
	return Ix, Iy

def sobelKernel():
	sobelx = np.zeros((3, 3), int)
	sobely = np.zeros((3, 3), int)

 	sobelx[0, 0] = 1
	sobelx[0, 1] = 2
	sobelx[0, 2] = 1
	sobelx[2, 0] = -1
	sobelx[2, 1] = -2
	sobelx[2, 2] = -1

	sobely[0, 0] = 1
	sobely[1, 0] = 2
	sobely[2, 0] = 1
	sobely[0, 2] = -1
	sobely[1, 2] = -2
	sobely[2, 2] = -1

	return sobelx, sobely

def GaussianFilter(sigma):
    halfSize = 3 * sigma
    maskSize = 2 * halfSize + 1 
    mat = np.ones((maskSize,maskSize)) / (float)( 2 * np.pi * (sigma**2))
    xyRange = np.arange(-halfSize, halfSize+1)
    xx, yy = np.meshgrid(xyRange, xyRange)    
    x2y2 = (xx**2 + yy**2)    
    exp_part = np.exp(-(x2y2/(2.0*(sigma**2))))
    mat = mat * exp_part
    
    # plotting this function
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(xx, yy, mat, rstride=1, cstride=1, cmap='hot')
    plt.show()   
    cv2.waitKey()

    return mat



def my_MagAndOrientation(Ix, Iy, t_low):

	print(Ix.shape)
	sizex, sizey = Ix.shape
	magnitude = np.zeros((sizex, sizey), float)
	orientation = np.zeros((sizex, sizey), float)
	auxValue = np.arctan2(Iy, Ix)
	print(auxValue[0][0])
	for index1, (x, y) in enumerate(zip(Ix,Iy)):
		for index2, (xValue, yValue) in enumerate(zip(x, y)):
			preSquareRoot = math.pow(xValue, 2) + math.pow(yValue, 2)
			magnitude[index1][index2] = math.sqrt(preSquareRoot)
	
			if auxValue[index1][index2] > (-math.pi/8) and auxValue[index1][index2] < (math.pi*1/8) or auxValue[index1][index2] < (-math.pi*7/8) or auxValue[index1][index2] > (math.pi*7/8): # and auxValue[index1][index2] != 0.0:	
				orientation[index1][index2] = 0
			elif auxValue[index1][index2] > (math.pi*1/8) and auxValue[index1][index2] < (math.pi*3/8) or auxValue[index1][index2] > (-math.pi*7/8) and auxValue[index1][index2] < (-math.pi*5/8):
				orientation[index1][index2] = 3
			elif auxValue[index1][index2] > (math.pi*3/8) and auxValue[index1][index2] < (math.pi*5/8) or auxValue[index1][index2] < (-math.pi*3/8) and auxValue[index1][index2] > (-math.pi*5/8):
				orientation[index1][index2] = 2
			elif auxValue[index1][index2] > (math.pi*5/8) and auxValue[index1][index2] < (math.pi*7/8) or auxValue[index1][index2] > (-math.pi*3/8) and auxValue[index1][index2] < (-math.pi*1/8):
				orientation[index1][index2] = 1
			else:
				print(auxValue[index1][index2])
			

	magnitudeNormalized = cv2.normalize(magnitude.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	
	cv2.imshow('MagnitudeNormalized', magnitudeNormalized)
	cv2.imshow('Orientation', orientation)
	cv2.waitKey()
	return magnitudeNormalized, orientation

def my_NMS(mag, orient, t_low):
	mag_thin = np.zeros(mag.shape, float)

	for i, (x, y) in enumerate(zip(mag, orient)):
		for j, (xValue, yValue) in enumerate(zip(x, y)):
	
			if mag[i][j] > t_low:

				if orient[i][j] == 0:
					if mag[i-1, j] < mag[i][j] and mag[i+1, j] < mag[i][j]:
						mag_thin[i][j] = mag[i][j] 
						
				elif orient[i][j] == 1:
					if mag[i+1, j-1] < mag[i][j] and mag[i-1, j+1] < mag[i][j]:
						mag_thin[i][j] = mag[i][j]
				elif orient[i][j] == 2:
					if mag[i, j-1] < mag[i][j] and mag[i, j+1] < mag[i][j]:
						mag_thin[i][j] = mag[i][j]
				elif orient[i][j] == 3:
					if mag[i-1, j-1] < mag[i][j] and mag[i+1, j+1] < mag[i][j]:
						mag_thin[i][j] = mag[i][j]
				else:
					print(orient[i][j])

			
	cv2.imshow('MagThin', mag_thin)
	cv2.waitKey()
	return mag_thin

def my_linking(mag_thin, orient, tLow, tHigh):
	result_binary = np.zeros(mag_thin.shape, float)

	for i, (x, y) in enumerate(zip(mag_thin, orient)):
		for j, (xValue, yValue) in enumerate(zip(x, y)):

			if mag_thin[i][j] >= tHigh:
				#ForwardScan
				if orient[i][j] == 0:
					if mag_thin[i+1, j] > tLow:
						mag_thin[i+1, j] = tHigh 
				elif orient[i][j] == 1:
					if mag_thin[i-1, j+1] > tLow:
						mag_thin[i-1, j+1] = tHigh
				elif orient[i][j] == 2:
					if mag_thin[i, j+1] > tLow:
						mag_thin[i, j+1] = tHigh
				elif orient[i][j] == 3:
					if mag_thin[i+1, j+1] > tLow:
						mag_thin[i+1, j+1] = tHigh
	
	for i, (x, y) in enumerate(zip(reversed(mag_thin), reversed(orient))):
		for j, (xValue, yValue) in enumerate(zip(reversed(x), reversed(y))):

			if mag_thin[i][j] >= tHigh:

				#BackwardScan
				if orient[i][j] == 0:
					if mag_thin[i-1, j] > tLow:
						mag_thin[i-1, j] = tHigh 
				elif orient[i][j] == 1:
					if mag_thin[i+1, j-1] > tLow:
						mag_thin[i+1, j-1] = tHigh
				elif orient[i][j] == 2:
					if mag_thin[i, j-1] > tLow:
						mag_thin[i, j-1] = tHigh
				elif orient[i][j] == 3:
					if mag_thin[i-1, j-1] > tLow:
						mag_thin[i-1, j-1] = tHigh


	for i, x in enumerate(mag_thin):
		for j, y in enumerate(x):
			if mag_thin[i, j] >= tHigh:
				result_binary[i, j] = 1

	cv2.imshow('Result Binary', result_binary)
	cv2.waitKey()

	return result_binary


def my_Canny(img, sigma, tLow, tHigh):
	firstStep = my_Normalize(testImage)
	cv2.imshow("Before", firstStep)
	final_ix, final_iy = my_DerivativesOfGaussian(firstStep, sigma)
	mag, orient = my_MagAndOrientation(final_ix, final_iy, tLow)
	mag_thin = my_NMS(mag, orient, tLow)

	final_resultBinary = my_linking(mag_thin, orient, tHigh, tLow)	

	cv2.destroyAllWindows()


testImage = cv2.imread('testImages/TestImg2_Signs.jpg')

my_Canny(testImage, 1, 0.08, 0.8)


