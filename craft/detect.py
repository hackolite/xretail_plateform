import matplotlib.pyplot as plt
import cv2
import keras_ocr
import PIL
from PIL import Image
import os
import time



import random
import string

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    print("Random string of length", length, "is:", result_str)



#mode crop==False retourne l'image entière avec les coordonnées boxes relative a l'image 

def model_constructor():
    pass



def execute(model_path=None, image=None, config=None, crop=False, draw=False):
    """ l'image ressort intact """
    detector = keras_ocr.detection.Detector()
    if model_path != None:
        detector.model.load_weights(model_path)
    else:
        detector = keras_ocr.detection.Detector(weights='clovaai_general')
    boxes = detector.detect(images=[image])
    
    if crop == False:
        for box in boxes:
            for i in box:
                cv2.rectangle(image,(i[0],i[1]),(i[2],i[3]),(0,255,0),3)
        #cv2.imwrite("./{0}.jpeg".format(get_random_string(7)), image)
        return {"image":image, "boxes":boxes}

