from ultralytics import YOLO
import cv2


def model_constructor():
	pass


def execute(model_path=None, image=None, config=None):
	model = YOLO(model_path)
	results = model(image)
	boxes = results[0].boxes.xyxy.tolist()
	return {"image":image, "boxes":boxes}
