import time
import cv2
import mss
import numpy as np
from matplotlib import pyplot as plt

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

with mss.mss() as sct:
    # Part of the screen to capture
    monitor = {'top': 40, 'left': 0, 'width': 800, 'height': 640}

    while(True):
        last_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        img = np.array(sct.grab(monitor))

        # Process the image and display it
	img = image_processing(img)
	cv2.imshow('screenshots',img)	
	
        print('FPS: {0}'.format(1 / (time.time()-last_time)))

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
