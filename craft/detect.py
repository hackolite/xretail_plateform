import matplotlib.pyplot as plt
import cv2
import keras_ocr
import PIL
from PIL import Image
import os
import time



def model_constructor():
    pass


def execute(model_path=None, image=None, config=None):
    detector = keras_ocr.detection.Detector()
    detector.model.load_weights(model_path)
    boxes = detector.detect(images=[image])[0]
    return {"image":image, "boxes":boxes}