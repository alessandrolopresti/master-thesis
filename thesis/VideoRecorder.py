from __future__ import print_function, division
import numpy as np
import cv2
import threading
import time
import settings
from FacialEmotionRecognitionModel import FacialEmotionRecognitionModel

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialEmotionRecognitionModel("model.json", "weights.h5")


class VideoRecorder():
    "Video class based on openCV"

    def __init__(self, name="temp_video.avi", fourcc="MJPG", sizex=640, sizey=480, camindex=0, fps=17,
                 capture_duration=5):
        self.capture_duration = capture_duration
        self.device_index = camindex
        self.emotion = []
        self.fps = fps  # fps should be the minimum constant rate at which the camera can
        self.fourcc = fourcc  # capture images (with no decrease in speed over time; testing is required)
        self.frameSize = (sizex, sizey)  # video formats and sizes also depend and vary according to the camera used
        self.video_filename = name
        self.video_cap = cv2.VideoCapture(self.device_index)
        self.video_writer = cv2.VideoWriter_fourcc(*self.fourcc)
        self.video_out = cv2.VideoWriter(self.video_filename, self.video_writer, self.fps, self.frameSize)
        self.frame_counts = 1
        self.start_time = time.time()

    def record(self):
        start_time = time.time()
        while (int(time.time() - start_time) < self.capture_duration):
            ret, video_frame = self.video_cap.read()

            gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.3, 5)
            cv2.imshow('Facial Emotion Recognition', video_frame)
            if (len(faces) > 0):
                for (x, y, w, h) in faces:
                    fc = gray[y:y + h, x:x + w]
                    roi = cv2.resize(fc, (48, 48))
                    videoPrediction = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
                    self.emotion.append(videoPrediction)

            if ret:
                self.video_out.write(video_frame)
                self.frame_counts += 1
            else:
                self.i += 1
                break

        self.stop()

    def stop(self):
        "Finishes the video recording therefore the thread too"
        self.video_out.release()
        self.video_cap.release()
        cv2.destroyAllWindows()
        if (len(self.emotion) > 0):
            settings.video_emotion = max(set(self.emotion), key=self.emotion.count)

    def start(self):
        "Launches the video recording function using a thread"
        video_thread = threading.Thread(target=self.record)
        video_thread.start()
        return video_thread