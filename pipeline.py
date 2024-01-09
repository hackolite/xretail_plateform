from retinanet import detect as retinanet_detect
from cropad import detect as croppad_detect
from craft import detect as craft_detect
import cv2
import os 


folder_dataset = "./dataset"


#crop_pad
#price_pad
#


import random
import string


def draw_rectangle():
	pass





def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str



def filter(filter="larger"):
	pass

def walk_path(folder_path):
	#recognize text for price thanks to trOcr	
	for top, dirs, files in os.walk(folder_dataset):
		for file in files:
			file = os.path.join(top, file)
			im = cv2.imread(file)
			yield im	


def area(box):
	return (box[2] - box[0]) * (box[3] - box[1])


def execute(image_cv=None):
	#croppad
	box_max = [0,0,0,0]
	resultat_yolo = croppad_detect.execute(model_path="/home/lamaaz/xretail_plateform/models/yolo/pad.pt", image=image_cv, config=None)
	if len(resultat_yolo["boxes"]) > 0:
		boxes = resultat_yolo["boxes"]
		for box in boxes:
				box = [int(i) for i in box]
				cv2.rectangle(image_cv,(box[0],box[1]),(box[2],box[3]),(0,255,255),3)
				if area(box) > area(box_max):
					box_max = box

		pad = image_cv[box[1]:box[3], box[0]:box[2]]
	else:
		pad = image_cv 
	

	#crop price, select larger area
	resultat_retina = retinanet_detect.execute(model_path="./models/retinanet/retina_price_4.pt", image=pad, config=None)

	if len(resultat_retina["boxes"]) > 0:
		boxes = resultat_retina["boxes"]
		for box in boxes:
				box = [int(i) for i in box]
				cv2.rectangle(pad,(box[0],box[1]),(box[2],box[3]),(0,0,255),13)

	


	#detect text on whole pad area, and extract EAN, which is the wider box
	resultat_craft = craft_detect.execute(model_path=None, image=im, config=None)
	for i in resultat_craft["boxes"]:
		print("boxe :", i)

	#cv2.imwrite("./{0}_retina_craft.jpeg".format(get_random_string(7)), pad)



def scan(folder=None):
	gen = walk_path(folder_dataset)
	for im in gen:
		resultat = execute(im)


if __name__ == "__main__":
	scan(folder_dataset)
