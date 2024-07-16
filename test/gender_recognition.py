import torch
import requests
from PIL import Image
from io import BytesIO
import numpy as np

from transformers import AutoImageProcessor, AutoModelForImageClassification

# # Get example image from official fairface repo + read it in as an image
# r = requests.get('https://github.com/dchen236/FairFace/blob/master/detected_faces/race_Asian_face0.jpg?raw=true')
# im = Image.open(BytesIO(r.content))

# # save image
# im.save("person_image/18.jpg")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

image_path = f"person_image/12.jpg"
image = Image.open(image_path)
image_array = np.array(image)

# # Init model, transforms
# model = AutoModelForImageClassification.from_pretrained('rizvandwiki/gender-classification-2')
# image_processor = AutoImageProcessor.from_pretrained('rizvandwiki/gender-classification-2')

model_path = "/root/lifang535/nsl_project/efficiency_attack/multi-model_application/dance/model/gender-classification-2/gender-classification-2_model.pth"
image_processor_path = "/root/lifang535/nsl_project/efficiency_attack/multi-model_application/dance/model/gender-classification-2/gender-classification-2_image_processor.pth"

# torch.save(model, model_path)
# torch.save(image_processor, image_processor_path)

model = torch.load(model_path, map_location=device)
image_processor = torch.load(image_processor_path)

# Transform our image and pass it through the model
# inputs = image_processor(image, return_tensors='pt').to(device)
inputs = image_processor(images=[image_array], return_tensors="pt").to(device)

# inputs = { # 错误的写法
#     'pixel_values': (torch.from_numpy(image_array.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0).to(device)
# }
with torch.no_grad():
    # output = model(**inputs)
    output = model(inputs['pixel_values'])

# Predicted Class probabilities
proba = output.logits.softmax(1)

# Predicted Classes
preds = proba.argmax(1)

gender = model.config.id2label[preds.item()]

print(f"Predicted Gender: {gender}")

