import os
import cv2
import time
import torch

import numpy as np

from PIL import Image
from queue import Empty
from threading import Thread
from multiprocessing import Process, Queue
from concurrent.futures import ThreadPoolExecutor

from configs import config
from request import Request

class GenderRecognition(Process):
    def __init__(self, 
                 config: dict,
                 person_frame_queue: Queue,
                 gender2age_queue: Queue,):
        super().__init__()

        # from person_recognition module
        self.person_frame_queue = person_frame_queue
        # to age_recognition module
        self.gender2age_queue = gender2age_queue
        
        self.frame_size = config['frame_size']
        
        self.device = torch.device(config['gender_recognition']['device'])
        self.image_processor_path = config['gender_recognition']['image_processor_path']
        self.model_path = config['gender_recognition']['model_path']
        
        self.image_processor = None
        self.model = None
        self.id2label = None
        
        # self.thread_pool = ThreadPoolExecutor(max_workers=1000)
        
        self.end_flag = False

    def run(self):
        print(f"[GenderRecognition] Start!")
        
        self.image_processor = torch.load(self.image_processor_path, map_location=self.device)
        self.model = torch.load(self.model_path, map_location=self.device)
        self.id2label = self.model.config.id2label
        
        while not self.end_flag:
            try:
                request = self.person_frame_queue.get(timeout=1)
            except Empty:
                continue
            
            if request is None:
                self._end()
                break
            
            # temp_thread = self.thread_pool.submit(self._infer, request)
            
            self._infer(request)
    
    def _infer(self, request):
        if request.box is not None:
            frame_array = request.data
            
            frame_array = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
            
            frame_size = frame_array.shape
            
            box = request.box
            
            # Relative coordinates need to be converted to absolute coordinates
            x1, y1, x2, y2 = box
            x1 = int(x1 * frame_size[1])
            y1 = int(y1 * frame_size[0])
            x2 = int(x2 * frame_size[1])
            y2 = int(y2 * frame_size[0])
            
            frame_array_cropped = frame_array[y1:y2, x1:x2]
            
            inputs = self.image_processor(images=[frame_array_cropped], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                # outputs = self.model(**inputs)
                outputs = self.model(inputs['pixel_values'])
                
            # Predicted Class probabilities
            proba = outputs.logits.softmax(1)
            # Predicted Classes
            preds = proba.argmax(1)
            # Predicted Gender
            gender = self.id2label[preds.item()]
            
            label = gender
            
            request.label += f" {label}"
        
        self.gender2age_queue.put(request)
            
    def _end(self):
        self.gender2age_queue.put(None)
        
        self.end_flag = True
        print(f"[GenderRecognition] Stop!")

        
if __name__ == '__main__':
    person_frame_queue = Queue()
    gender2age_queue = Queue()
    gender_recognition = GenderRecognition(config, person_frame_queue, gender2age_queue)
    gender_recognition.start()
    try:
        gender_recognition.join()
    except KeyboardInterrupt:
        print("[main] KeyboardInterrupt")
        gender_recognition.terminate()
        