import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2
import os
import time

#loading the data


starting_value = 1
train_data = []

while True:
	file_name = '/home/nayangupta824/python_drives/Data/training_data-{}.npy'.format(starting_value)

	if os.path.isfile(file_name):
		print('Loading file ',starting_value)
		tmp = np.load(file_name)
		for d in tmp:
			train_data.append(d)
		starting_value += 1
	else:
		break

#print data statistics
print 'Data size :', len(train_data)

#to add number to rows
df = pd.DataFrame(train_data)

#print count of left,right,null
print(Counter(df[1].apply(str)))


print 'Training data before shuffling ...... '

time.sleep(2)

for data in train_data:
	img = data[0]
	choice = data[1]
	img = cv2.resize(img,(200,200))
	cv2.imshow('test',img)
	print choice
	if cv2.waitKey(25) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break
		
		
print 'Training data after shuffling ...... '
shuffle(train_data)
time.sleep(2)

for data in train_data:
	img = data[0]
	choice = data[1]
	img = cv2.resize(img,(200,200))
	cv2.imshow('test',img)
	print choice
	if cv2.waitKey(25) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break
		
#save the shuffled data
np.save('/home/nayangupta824/python_drives/Final_training_data/training_data_final.npy',train_data)
