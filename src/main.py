import os
import threading
import time
import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Ruta al modelo relativa a este archivo (funciona sin importar desde donde se ejecute)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models', 'hand_landmarker.task')

# Mano a detectar: 'Right', 'Left' o None (ambas)
TARGET_HAND = 'Right'


def corrected_label(raw_label):
    """El frame se voltea con cv.flip (espejo), lo que invierte la lateralidad
    que reporta MediaPipe; por eso se intercambian las etiquetas."""
    return {'Left': 'Right', 'Right': 'Left'}.get(raw_label, raw_label)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),          # indice
    (5, 9), (9, 10), (10, 11), (11, 12),     # medio
    (9, 13), (13, 14), (14, 15), (15, 16),   # anular
    (13, 17), (17, 18), (18, 19), (19, 20),  # menique
    (0, 17),                                 # palma
]

# El callback se ejecuta en otro hilo: guardamos el ultimo resultado con un lock
latest_result = None
result_lock = threading.Lock()


def save_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    with result_lock:
        latest_result = result


def draw_hand_graph(frame, result):
    """Dibuja los landmarks (nodos) y conexiones (aristas) de cada mano detectada."""
    if result is None or not result.hand_landmarks:
        return frame
    h, w = frame.shape[:2]
    for i, hand in enumerate(result.hand_landmarks):
        label = None
        if result.handedness and i < len(result.handedness):
            label = corrected_label(result.handedness[i][0].category_name)
        # Filtrar: solo se dibuja la mano elegida en TARGET_HAND
        if TARGET_HAND is not None and label != TARGET_HAND:
            continue
        # Coordenadas normalizadas [0,1] -> pixeles
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for a, b in HAND_CONNECTIONS:
            cv.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for p in pts:
            cv.circle(frame, p, 4, (0, 0, 255), -1)
        # Etiqueta de mano (Left/Right)
        if label is not None:
            cv.putText(frame, label, (pts[0][0], pts[0][1] + 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return frame


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_DIR),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,  # se detectan 2 y luego se filtra por TARGET_HAND
    result_callback=save_result)  # se pasa la funcion, NO se llama

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("no se pudo abrir la camara")
        exit()

    last_ts = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            print("no se pudo recibir el frame (acabado el streaming?). saliendo")
            break
        frame = cv.flip(frame,1)
        frame2RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame2RGB)

        timestamp_ms = int(time.monotonic() * 1000)
        if timestamp_ms <= last_ts:
            timestamp_ms = last_ts + 1
        last_ts = timestamp_ms
        landmarker.detect_async(mp_image, timestamp_ms)

        with result_lock:
            result = latest_result
        draw_hand_graph(frame, result)

        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()