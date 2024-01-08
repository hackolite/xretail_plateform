import matplotlib.pyplot as plt
import cv2
import keras_ocr
import PIL
from PIL import Image
import os
import time

#mode crop==False retourne l'image entière avec les coordonnées boxes relative a l'image 

def model_constructor():
    pass



def execute(model_path=None, image=None, config=None, crop=False):
    detector = keras_ocr.detection.Detector()
    if model_path != None:
        detector.model.load_weights(model_path)
    else:
        detector = keras_ocr.detection.Detector(weights='clovaai_general')
    boxes = detector.detect(images=[image])[0]
    if crop == False:
        return {"image":image, "boxes":boxes}
    else:
        pass