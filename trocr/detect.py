from transformers import VisionEncoderDecoderModel
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image




def execute(image):
	image_cv = image
	device = "cuda" if torch.cuda.is_available() else "cpu"
	processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
	model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
	model.to(device)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	ixel_values = processor(images=image_cv, return_tensors="pt").pixel_values
	generated_ids = model.generate(pixel_values)
	generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
	return generated_text