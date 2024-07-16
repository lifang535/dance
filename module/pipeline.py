from multiprocessing import Process, Queue

from configs import config

from video_to_frame import VideoToFrame
from object_detection import ObjectDetection
from posture_recognition import PostureRecognition
from person_recognition import PersonRecognition
from gender_recognition import GenderRecognition
from age_recognition import AgeRecognition
from expression_recognition import ExpressionRecognition
from frame_to_video import FrameToVideo

#---------------------------------------------------------------------------
# Create a traffic monitoring pipeline
#---------------------------------------------------------------------------
if __name__ == '__main__':
    frame_queue = Queue()
    posture_queue = Queue()
    person_queue = Queue()
    posture_frame_queue = Queue()
    person_frame_queue = Queue()
    
    gender2age_queue = Queue()
    age2expression_queue = Queue()
    expression_queue = Queue()
    
    video_to_frame = VideoToFrame(config, frame_queue)
    object_detection = ObjectDetection(config, frame_queue, posture_queue, person_queue)
    posture_recognition = PostureRecognition(config, posture_queue, posture_frame_queue)
    # posture_recognition_2 = PostureRecognition(config, posture_queue, posture_frame_queue) # TODO: 需要把结束信号放回到队列中
    person_recognition = PersonRecognition(config, person_queue, person_frame_queue)
    gender_recognition = GenderRecognition(config, person_frame_queue, gender2age_queue)
    age_recognition = AgeRecognition(config, gender2age_queue, age2expression_queue)
    expression_recognition = ExpressionRecognition(config, age2expression_queue, expression_queue)
    frame_to_video = FrameToVideo(config, posture_frame_queue, expression_queue)
    
    video_to_frame.start()
    object_detection.start()
    posture_recognition.start()
    person_recognition.start()
    gender_recognition.start()
    age_recognition.start()
    expression_recognition.start()
    frame_to_video.start()
    
    try:
        video_to_frame.join()
        object_detection.join()
        posture_recognition.join()
        person_recognition.join()
        gender_recognition.join()
        age_recognition.join()
        expression_recognition.join()
        frame_to_video.join()
    except KeyboardInterrupt:
        print("[main] KeyboardInterrupt")
        video_to_frame.terminate()
        object_detection.terminate()
        posture_recognition.terminate()
        person_recognition.terminate()
        gender_recognition.terminate()
        age_recognition.terminate()
        expression_recognition.terminate()
        frame_to_video.terminate()

    print("[main] Pipeline end!")

