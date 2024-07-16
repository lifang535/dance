import os
import cv2
import time
import torch

from queue import Empty
from multiprocessing import Process, Queue

from configs import config
from request import Request

class PostureRecognition(Process):
    def __init__(self, 
                 config: dict,
                 posture_queue: Queue,
                 posture_frame_queue: Queue):
        super().__init__()
        
        # from object_detection module
        self.posture_queue = posture_queue
        # to frame_to_video module
        self.posture_frame_queue = posture_frame_queue
        
        self.device = torch.device(config['posture_recognition']['device'])
        self.image_processor_path = config['posture_recognition']['image_processor_path']
        self.model_path = config['posture_recognition']['model_path']
        
        self.image_processor = None
        self.model = None
        
        # self.thread_pool = ThreadPoolExecutor(max_workers=1000)
        
        self.end_flag = False

    def run(self):
        print(f"[PostureRecognition] Start!")
        
        self.model = torch.load(self.model_path, map_location=self.device)
        
        while not self.end_flag:
            try:
                request = self.posture_queue.get(timeout=1)
            except Empty:
                continue
            
            if request is None:
                self._end()
                break
            
            self._infer(request)
            
            # print(f"[PostureRecognition] video_id: {request.video_id}, frame_id: {request.frame_id}, posture_id: {request.posture_id}")
            
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
            
            inputs = frame_array[y1:y2, x1:x2]

            with torch.no_grad():
                try: # add
                    candidate, subset = self.model(inputs)
                except Exception as e:
                    print(f"[PostureRecognition] Error: {e}")
                    print(f"[PostureRecognition] inputs: {inputs.shape}")
                    candidate, subset = [], []
                
            request.candidate = candidate
            request.subset = subset
            
            # print(f"[PostureRecognition] video_id: {request.video_id}, frame_id: {request.frame_id}, posture_id: {request.posture_id}, label: {request.label}")
            
        self.posture_frame_queue.put(request)
        pass
            
    def _end(self):
        self.posture_frame_queue.put(None)
        
        self.end_flag = True
        print(f"[PostureRecognition] Stop!")
