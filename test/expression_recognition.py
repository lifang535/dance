import torch
import requests
from PIL import Image
from io import BytesIO

from transformers import AutoImageProcessor, AutoModelForImageClassification

# # Get example image from official fairface repo + read it in as an image
# r = requests.get('https://github.com/dchen236/FairFace/blob/master/detected_faces/race_Asian_face0.jpg?raw=true')
# im = Image.open(BytesIO(r.content))

# # save image
# im.save("person_image/18.jpg")

image_path = f"person_image/22.jpg"
image = Image.open(image_path)

# # Init model, transforms
# model = AutoModelForImageClassification.from_pretrained('trpakov/vit-face-expression')
# image_processor = AutoImageProcessor.from_pretrained('trpakov/vit-face-expression')

model_path = "/root/lifang535/nsl_project/efficiency_attack/multi-model_application/dance/model/vit-face-expression/vit-face-expression_model.pth"
image_processor_path = "/root/lifang535/nsl_project/efficiency_attack/multi-model_application/dance/model/vit-face-expression/vit-face-expression_image_processor.pth"

# torch.save(model, model_path)
# torch.save(image_processor, image_processor_path)

model = torch.load(model_path)
image_processor = torch.load(image_processor_path)

# Transform our image and pass it through the model
inputs = image_processor(image, return_tensors='pt')
with torch.no_grad():
    output = model(**inputs)

# Predicted Class probabilities
proba = output.logits.softmax(1)

# Predicted Classes
preds = proba.argmax(1)

expression = model.config.id2label[preds.item()]

print(f"Predicted Expression: {expression}")

