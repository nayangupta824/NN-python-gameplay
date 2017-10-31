from __future__ import print_function
import time
import cv2
import mss
import numpy as np
from matplotlib import pyplot as plt
import os
import pyxhook
import time


w    = [1,0,0,0,0,0,0,0,0]
s    = [0,1,0,0,0,0,0,0,0]
a    = [0,0,1,0,0,0,0,0,0]
d    = [0,0,0,1,0,0,0,0,0]
wa   = [0,0,0,0,1,0,0,0,0]
wd   = [0,0,0,0,0,1,0,0,0]
sa   = [0,0,0,0,0,0,1,0,0]
sd   = [0,0,0,0,0,0,0,1,0]
null = [0,0,0,0,0,0,0,0,1]

starting_value = 1
keys = []

def kbevent(event):
    global running
    # print key info
    key = event.Key
    keys.append(key)

# Create hookmanager

hookman = pyxhook.HookManager()
# Define our callback to fire when a key is pressed down
hookman.KeyDown = kbevent
# Hook the keyboard
hookman.HookKeyboard()
# Start our listener
hookman.start()


while True:
    file_name = 'training_data-{}.npy'.format(starting_value)

    if os.path.isfile(file_name):
        print('File exists, moving along',starting_value)
        starting_value += 1
    else:
        print('File does not exist, starting fresh!',starting_value)

        break

# This function is called every time a key is presssed

def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    '''
    output = [0,0,0,0,0,0,0,0,0]

    if 'w' in keys and 'a' in keys:
        output = wa
    elif 'w' in keys and 'd' in keys:
        output = wd
    elif 's' in keys and 'a' in keys:
        output = sa
    elif 's' in keys and 'd' in keys:
        output = sd
    elif 'w' in keys:
        output = w
    elif 's' in keys:
        output = s
    elif 'a' in keys:
        output = a
    elif 'd' in keys:
    	output = d
    else:
        output = null
    return output

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
    vertices = np.array([[10,540],[10,480],[800,480],[800,540],
                         ], np.int32)

    processed_img = roi(processed_img, [vertices])

    return processed_img

def main(file_name, starting_value):
    file_name = file_name
    starting_value = starting_value
    training_data = []
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    with mss.mss() as sct:
        # Part of the screen to capture
        monitor = {'top': 40, 'left': 40, 'width': 640, 'height': 480}

        while(True):
            last_time = time.time()
            global keys
            keys = []
            # Get raw pixels from the screen, save it to a Numpy array
            img = np.array(sct.grab(monitor))

            # Process the image and display
            img = image_processing(img)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow('screenshots',img)
            print("Array: ",keys)
            output = keys_to_output(keys)
            print("Array: ",keys)
            print('FPS: {0}'.format(1 / (time.time()-last_time)))
            training_data.append([img,output])
            
            if len(training_data) % 100 == 0:
                print(len(training_data))
            
                if len(training_data) == 500:
                    np.save(file_name,training_data)
                    print('SAVED')
                    training_data = []
                    starting_value += 1
                    file_name = '/home/nayangupta824/training_data-{}.npy'.format(starting_value)
            # Press "q" to quit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

main(file_name, starting_value)
