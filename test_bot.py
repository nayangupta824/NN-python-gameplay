from __future__ import print_function
import time
import cv2
import mss
import numpy as np
from matplotlib import pyplot as plt
import keyboard
import os
from alexnet import alexnet
from subprocess import Popen, PIPE

WIDTH = 60
HEIGHT = 80
#learning rate
LR = 1e-3
EPOCHS = 8
MODEL_NAME = 'python-drives-{}-{}-{}-epochs.model'.format(LR,'alexnetv2',EPOCHS)

w_down = '''keydown w
'''
w_up = '''keyup w
'''
s_down = '''keydown s
'''
s_up = '''keyup s
'''
a_down = '''keydown a
'''
a_up = '''keyup a
'''
d_down = '''keydown d
'''
d_up = '''keyup d
'''

def keypress(sequence):
    p = Popen(['xte'], stdin=PIPE)
    p.communicate(input=sequence)

def left():
	keypress(w_up)
	keypress(d_up)
	keypress(a_down)

def right():
	keypress(w_up)
	keypress(a_up)
	keypress(d_down)
	
def null():
	keypress(w_down)
	keypress(a_up)
	keypress(d_up)

def roi(img, vertices):

	#blank mask:
	mask = np.zeros_like(img)

	#filling pixels inside the polygon defined by "vertices" with the fill color
	cv2.fillPoly(mask, vertices, 255)

	#returning the image only where mask pixels are nonzero
	masked = cv2.bitwise_and(img, mask)
	return masked

def image_processing(image):
	original_image = image
	# convert to gray
	processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# edge detection
	processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300)

	processed_img = cv2.GaussianBlur(processed_img,(5,5),0)
	vertices = np.array([[70,400],[70,280],[600,280],[600,400],
						 ], np.int32)

	processed_img = roi(processed_img, [vertices])

	return processed_img
	
def main():
	print('Loading Model .....')
	model = alexnet(WIDTH, HEIGHT, LR)
	model.load('/home/nayangupta824/python_drives/model_old1/'+MODEL_NAME)
	for i in list(range(4))[::-1]:
        	print(i+1)
        	time.sleep(1)
	with mss.mss() as sct:
		# Part of the screen to capture
		monitor = {'top': 100, 'left': 0, 'width': 640, 'height': 480}

		while(True):
			img = np.array(sct.grab(monitor))
			# Process the image and display
			img = image_processing(img)
			img = cv2.resize(img,(80,60))
			img = img.reshape(60,80,1)
			
			prediction = model.predict([img])[0]
			#keypress(w_down)
			if prediction[0] > 0.85:
				left()
				print('left')
			elif prediction[1] > 0.95:
				right()
				print('right')
			else:
				null()
				print('null')
			#print(prediction)


main()
