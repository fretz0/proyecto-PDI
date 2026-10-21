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

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def corrected_label(raw_label):
    """El frame se voltea con cv.flip (espejo), lo que invierte la lateralidad
    que reporta MediaPipe; por eso se intercambian las etiquetas."""
    return {'Left': 'Right', 'Right': 'Left'}.get(raw_label, raw_label)


#grafo
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


LANDMARK_NAMES = [
    'muneca',
    'pulgar_cmc', 'pulgar_mcp', 'pulgar_ip', 'pulgar_tip',
    'indice_mcp', 'indice_pip', 'indice_dip', 'indice_tip',
    'medio_mcp', 'medio_pip', 'medio_dip', 'medio_tip',
    'anular_mcp', 'anular_pip', 'anular_dip', 'anular_tip',
    'menique_mcp', 'menique_pip', 'menique_dip', 'menique_tip',
]


def get_hand_data(result):
    """Extrae los datos posicionales del grafo de la mano detectada.

    Devuelve una lista de dicts:
        {'label': 'Right'|'Left'|None, 'coords': np.ndarray de forma (21, 3)}
    donde cada fila es (x, y, z) del nodo i:
        x, y -> normalizados [0, 1] respecto al ancho/alto de la imagen
        z    -> profundidad relativa a la muneca (mas negativo = mas cerca de la camara)
    """
    hands = []
    if result is None or not result.hand_landmarks:
        return hands
    for i, hand in enumerate(result.hand_landmarks):
        label = None
        if result.handedness and i < len(result.handedness):
            label = corrected_label(result.handedness[i][0].category_name)
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
        hands.append({'label': label, 'coords': coords})
    return hands


def draw_hand_graph(frame, hands):
    """Dibuja los landmarks (nodos) y conexiones (aristas) de cada mano detectada."""
    h, w = frame.shape[:2]
    for hand in hands:
        # Coordenadas normalizadas [0,1] -> pixeles
        pts = [(int(x * w), int(y * h)) for x, y, _ in hand['coords']]
        for a, b in HAND_CONNECTIONS:
            cv.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for idx, p in enumerate(pts):
            cv.circle(frame, p, 4, (0, 0, 255), -1)
            # Indice del nodo junto al punto (para relacionarlo con el panel)
            cv.putText(frame, str(idx), (p[0] + 5, p[1] - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        # Etiqueta de mano (Left/Right)
        if hand['label'] is not None:
            cv.putText(frame, hand['label'], (pts[0][0], pts[0][1] + 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return frame


def draw_coords_panel(frame, hands):
    """Imprime en pantalla las coordenadas (x, y, z) de cada nodo del grafo."""
    if not hands:
        return frame
    line_h = 15
    font_scale = 0.4
    panel_w = 230
    for n, hand in enumerate(hands):
        x0 = 5 + n * (panel_w + 5)
        panel_h = line_h * (len(hand['coords']) + 2)
        # Fondo semitransparente para que el texto sea legible
        overlay = frame.copy()
        cv.rectangle(overlay, (x0, 5), (x0 + panel_w, 5 + panel_h), (0, 0, 0), -1)
        cv.addWeighted(overlay, 0.5, frame, 0.5, 0, dst=frame)

        y = 5 + line_h
        cv.putText(frame, f"{hand['label']}   x      y      z", (x0 + 5, y),
                   cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1)
        for idx, (x, yy, z) in enumerate(hand['coords']):
            y += line_h
            text = f"{idx:2d}: {x:6.3f} {yy:6.3f} {z:7.3f}"
            cv.putText(frame, text, (x0 + 5, y),
                       cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
    return frame


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_DIR),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.35, # Sensibilidad para encontrar la mano inicialmente
    min_hand_presence_confidence=0.30,  # Evita que se pierda cuando cambia de ángulo/pose
    min_tracking_confidence=0.30,       # Mantiene el rastreo en movimientos rápidos
    result_callback=save_result)  # se pasa la funcion, NO se llama

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv.VideoCapture(1) # cambiar camara 
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
        hands = get_hand_data(result)   # datos x, y, z de cada nodo  #falta normalizar
        print(f"{hands}\n")
        draw_hand_graph(frame, hands)
        draw_coords_panel(frame, hands)

        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()