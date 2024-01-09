from ultralytics import YOLO
import cv2

import random
import string


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def model_constructor():
	pass


def execute(model_path=None, image=None, config=None, crop=False):
	model = YOLO(model_path)
	results = model(image)
	boxes = results[0].boxes.xyxy.tolist()
	if crop == False:
		for box in boxes:
			box = [int(i) for i in box]
			try:
				cv2.rectangle(image,(box[0],box[1]),(box[2],box[3]),(0,255,0),10)
			except Exception as e:
				print(e)
		#cv2.imwrite("./{0}.jpeg".format(get_random_string(7)), image)

	return {"image":image, "boxes":boxes}
