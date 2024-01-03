from retinanet import detect as retinanet_detect
from cropad import detect as cropad_detect
import cv2


im = cv2.imread("/home/lamaaz/xretail_plateform/retinanet/zeycpsweoi.jpg")
resultat_yolo = cropad_detect.execute(model_path="/home/lamaaz/xretail_plateform/models/yolo/pad.pt", image=im, config=None)
print(resultat_yolo["boxes"])
resultat_retina = retinanet_detect.execute(model_path="./models/retinanet/retina_price_4.pt", image=im, config=None)
print(resultat_retina["boxes"])