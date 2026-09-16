import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time 

MODEL_DIR = 'models/hand_landmarker.task'


def print_result(result: mp.tasks.vision.HandLandmarkerResult , output_image: mp.Image, timestamp_ms: int):
    if result.hand_landmarks:
        print(f'mano detectada en : {timestamp_ms}')

def capturarVideo():
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_DIR),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result
        )

    with HandLandmarker.create_from_options(options) as landmarker:
        isRecording = 1
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("no se pudo abrir la camara")
            exit()

        while isRecording:
            ret,frame = cap.read()
            if not ret:
                print("no se pudo recibir el frame (acabado el streaming?). saliendo")
                isRecording = 0
                break

            frame2RGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB) # frame a trabajar
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame2RGB)
            timestamp_ms = int(time.time()*1000)
            landmarker.detect_async(mp_image,timestamp_ms)
            cv.imshow('frame',frame)
            if cv.waitKey(1) == ord('q'):
                isRecording = 0
        cap.release()
        cv.destroyAllWindows()



capturarVideo()