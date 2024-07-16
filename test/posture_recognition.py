import os
import cv2
import time
import torch
import threading
import multiprocessing
import numpy as np

import copy
from PIL import Image
# from src import model
# from src import util
# from src.body import Body
# from src.hand import Hand
import math

def draw_bodypose_with_shifting(canvas, candidate, subset, x0=0, y0=0):
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            cv2.circle(canvas, (int(x + x0), int(y + y0)), 4, colors[i], thickness=-1)
    
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0] + x0
            X = candidate[index.astype(int), 1] + y0
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    
    return canvas

image_id = 4

image_path = f"person_image/{image_id}.jpg"
image = Image.open(image_path)
image_array = np.array(image)
print(f"image_array.shape: {image_array.shape}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_dict_path = "/root/lifang535/nsl_project/efficiency_attack/multi-model_application/dance/model/pytorch-openpose/body_pose_model_dict.pth"
model_path = "/root/lifang535/nsl_project/efficiency_attack/multi-model_application/dance/model/pytorch-openpose/body_pose_model.pth"
image_processor_path = "/root/lifang535/nsl_project/efficiency_attack/multi-model_application/dance/model/pytorch-openpose/body_pose_image_processor.pth"

# model = Body(model_dict_path)
# image_processor = util.draw_bodypose_with_shifting

# torch.save(model, model_path)
# torch.save(image_processor, image_processor_path)

model = torch.load(model_path, map_location=device)
image_processor = torch.load(image_processor_path, map_location=device)

image = cv2.imread(image_path)

test_number = 100
start_time = time.time()
for i in range(test_number + 10):
    if i == 10:
        start_time = time.time()
    with torch.no_grad():
        candidate, subset = model(image_array)
end_time = time.time()

print(f"Average time: {round((end_time - start_time) / test_number, 4)}")
        

print(f"candidate: {candidate}")
print(f"subset: {subset}")

image = image_processor(image, candidate, subset, 50, 50)

output_path = f"output_image/{image_id}_pose.jpg"
cv2.imwrite(output_path, image)
