import argparse
import collections
import os
import time

# Suprimir logs ruidosos de TensorFlow/CUDA para ejecución limpia en CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import tensorflow as tf

# Rutas del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'gru_lsch_dinamico.h5')
HOLISTIC_TASK_PATH = os.path.join(BASE_DIR, '..', 'models', 'holistic_landmarker.task')

# 50 Clases del modelo dinámico de Lengua de Señas Chilena (LSCh)
ACTIONS = np.array([
    'NONE',
    # Alfabeto LSCh (incluye letras dinámicas G, J, Ñ, S, X, Z)
    'A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'P', 'R', 'T', 'U', 'V', 'W', 'Y', 'Q',
    'G', 'J', 'Ñ', 'S', 'X', 'Z',
    # Palabras y expresiones cotidianas LSCh
    'Hola', 'Chao', 'Comer', 'Caminar', 'Flaco', 'Alto', 'Bajo',
    'Bonito', 'Hablar', 'Ayudar', 'Abrir', 'Disculpar', 'Vender',
    'Dibujar', 'Planchar', 'Amigo', 'Responsable', 'Arrendar',
    'Chocar', 'Vestir', 'Estudiar', 'Jugar'
])

SEQUENCE_LENGTH = 30  # 30 frames temporales (~1 segundo)
NUM_FEATURES = 258     # Pose (132) + Mano Izq (63) + Mano Der (63)

# Conexiones anatómicas para graficar sobre el frame con OpenCV
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),          # Índice
    (5, 9), (9, 10), (10, 11), (11, 12),     # Medio
    (9, 13), (13, 14), (14, 15), (15, 16),   # Anular
    (13, 17), (17, 18), (18, 19), (19, 20),  # Meñique
    (0, 17)                                  # Palma
]

POSE_CONNECTIONS = [
    (11, 12),           # Hombros
    (11, 13), (13, 15), # Brazo izquierdo
    (12, 14), (14, 16)  # Brazo derecho
]


def load_dynamic_model(model_path):
    """Carga el modelo GRU con compatibilidad garantizada para Keras 2 y Keras 3."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

    from tensorflow.keras.layers import Dense, GRU, Input
    from tensorflow.keras.models import Sequential

    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
        GRU(64, return_sequences=True, activation='tanh'),
        GRU(64, return_sequences=False, activation='tanh'),
        Dense(len(ACTIONS), activation='softmax')
    ])
    model.load_weights(model_path)
    print(f"[OK] Modelo dinámico GRU cargado exitosamente desde {model_path}")
    return model


def extract_keypoints(result):
    """Extrae exactamente 258 características desde HolisticLandmarkerResult:

    - Pose: 33 puntos x 4 (x, y, z, visibilidad) = 132
    - Mano izquierda: 21 puntos x 3 (x, y, z) = 63
    - Mano derecha: 21 puntos x 3 (x, y, z) = 63
    """
    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        pose = np.array([
            [lm.x, lm.y, lm.z, getattr(lm, 'visibility', 0.0) or 0.0]
            for lm in result.pose_landmarks
        ], dtype=np.float32).flatten()
    else:
        pose = np.zeros(33 * 4, dtype=np.float32)

    if result.left_hand_landmarks and len(result.left_hand_landmarks) > 0:
        lh = np.array([
            [lm.x, lm.y, lm.z]
            for lm in result.left_hand_landmarks
        ], dtype=np.float32).flatten()
    else:
        lh = np.zeros(21 * 3, dtype=np.float32)

    if result.right_hand_landmarks and len(result.right_hand_landmarks) > 0:
        rh = np.array([
            [lm.x, lm.y, lm.z]
            for lm in result.right_hand_landmarks
        ], dtype=np.float32).flatten()
    else:
        rh = np.zeros(21 * 3, dtype=np.float32)

    return np.concatenate([pose, lh, rh])


def draw_landmarks(frame, result):
    """Dibuja con OpenCV los esqueletos de manos y extremidades superiores."""
    h, w = frame.shape[:2]

    # Dibujar Pose (Hombros, codos, muñecas)
    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in result.pose_landmarks]
        for a, b in POSE_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv.line(frame, pts[a], pts[b], (255, 128, 0), 2)
        for idx in [11, 12, 13, 14, 15, 16]:
            if idx < len(pts):
                cv.circle(frame, pts[idx], 4, (0, 165, 255), -1)

    # Dibujar Manos
    for hand_lms, color_line, color_dot in [
        (result.left_hand_landmarks, (255, 0, 128), (255, 255, 0)),
        (result.right_hand_landmarks, (0, 255, 128), (0, 255, 255))
    ]:
        if hand_lms and len(hand_lms) > 0:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
            for a, b in HAND_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv.line(frame, pts[a], pts[b], color_line, 2)
            for p in pts:
                cv.circle(frame, p, 3, color_dot, -1)


def draw_ui(frame, current_action, confidence, sentence, fps):
    """Despliega paneles semitransparentes, barra de confianza e historial."""
    h, w = frame.shape[:2]

    # Panel superior para predicción en vivo
    overlay = frame.copy()
    cv.rectangle(overlay, (0, 0), (w, 75), (20, 20, 20), -1)
    # Panel inferior para historial de señas
    cv.rectangle(overlay, (0, h - 50), (w, h), (15, 15, 15), -1)
    cv.addWeighted(overlay, 0.75, frame, 0.25, 0, dst=frame)

    if current_action == 'NONE':
        action_text = "Esperando seña..."
        tag_color = (150, 150, 150)
    else:
        action_text = f"Seña: {current_action}"
        tag_color = (0, 255, 128)

    # Texto de la seña actual
    cv.putText(frame, action_text, (20, 36),
               cv.FONT_HERSHEY_SIMPLEX, 0.85, tag_color, 2, cv.LINE_AA)

    # Barra gráfica de confianza
    bar_x = w - 300
    bar_w = 200
    bar_h = 16
    cv.rectangle(frame, (bar_x, 20), (bar_x + bar_w, 20 + bar_h), (80, 80, 80), 2)
    fill_w = int(bar_w * max(0.0, min(1.0, float(confidence))))
    cv.rectangle(frame, (bar_x, 20), (bar_x + fill_w, 20 + bar_h), (0, 220, 0), -1)
    cv.putText(frame, f"{int(confidence * 100)}%", (bar_x + bar_w + 10, 34),
               cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv.LINE_AA)

    # Estado de FPS y comandos
    cv.putText(frame, f"FPS: {fps:4.1f} | [q]: Salir  [c]: Limpiar historial", (20, 62),
               cv.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv.LINE_AA)

    # Historial de señas confirmadas
    history_str = "Historial: " + " ".join([f"[{s}]" for s in sentence[-8:]])
    cv.putText(frame, history_str, (20, h - 18),
               cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv.LINE_AA)


def open_webcam(preferred_idx=1):
    """Busca y abre la cámara especificada o la primera disponible."""
    candidates = [preferred_idx, 0, 2]
    seen = set()
    indices = [x for x in candidates if not (x in seen or seen.add(x))]

    for idx in indices:
        cap = cv.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"[OK] Cámara web conectada en índice {idx}")
                return cap, idx
            cap.release()
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Reconocimiento dinámico de LSCh en tiempo real")
    parser.add_argument('--camera', type=int, default=1, help="Índice preferido de cámara (default: 1)")
    parser.add_argument('--threshold', type=float, default=0.75, help="Umbral mínimo de confianza (default: 0.75)")
    args = parser.parse_args()

    # Cargar modelo clasificador
    model = load_dynamic_model(MODEL_PATH)

    # Configurar HolisticLandmarker con la API oficial de MediaPipe Tasks
    if not os.path.exists(HOLISTIC_TASK_PATH):
        raise FileNotFoundError(f"No se encontró el modelo de MediaPipe en: {HOLISTIC_TASK_PATH}")

    base_options = python.BaseOptions(model_asset_path=HOLISTIC_TASK_PATH)
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE
    )

    cap, active_cam = open_webcam(args.camera)
    if cap is None:
        print("[ERROR] No se pudo acceder a ninguna cámara disponible.")
        return

    sequence = collections.deque(maxlen=SEQUENCE_LENGTH)
    sentence = []
    current_action = 'NONE'
    confidence = 0.0

    # Lógica de confirmación (debounce) para el historial
    confirm_count = 0
    REQUIRED_CONSECUTIVE_FRAMES = 5
    last_confirmed_action = 'NONE'

    prev_time = time.time()
    fps = 0.0

    print("\n" + "=" * 62)
    print("  RECONOCIMIENTO DINÁMICO LSCh ACTIVO (50 CLASES)")
    print("  - Alfabeto completo A-Z (incluye dinámicas J, Z, G, S, X, Ñ)")
    print("  - Expresiones: Hola, Chao, Ayudar, Disculpar, Comer, etc.")
    print("  - Controles: [q] Salir  |  [c] Limpiar historial")
    print("=" * 62 + "\n")

    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[AVISO] Flujo de video finalizado.")
                break

            # Modo espejo y conversión para MediaPipe
            frame = cv.flip(frame, 1)
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Detección de landmarks con Tasks API
            result = landmarker.detect(mp_image)

            # Dibujar grafo sobre el frame
            draw_landmarks(frame, result)

            # Extraer vector de 258 características y guardar en buffer
            keypoints = extract_keypoints(result)
            sequence.append(keypoints)

            # Inferencia cuando se han acumulado los 30 frames
            if len(sequence) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(list(sequence), axis=0)
                pred = model(input_data, training=False).numpy()[0]
                best_idx = np.argmax(pred)
                pred_action = ACTIONS[best_idx]
                pred_conf = pred[best_idx]

                if pred_conf >= args.threshold:
                    current_action = pred_action
                    confidence = pred_conf

                    # Registrar en historial si la predicción es estable
                    if current_action == last_confirmed_action:
                        confirm_count += 1
                    else:
                        confirm_count = 1
                        last_confirmed_action = current_action

                    if confirm_count == REQUIRED_CONSECUTIVE_FRAMES and current_action != 'NONE':
                        if len(sentence) == 0 or sentence[-1] != current_action:
                            sentence.append(current_action)
                            print(f"[RECONOCIDO]: {current_action} ({int(confidence * 100)}%)")
                else:
                    current_action = 'NONE'
                    confidence = pred_conf
                    confirm_count = 0

            # Cálculo de FPS suavizado
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
            prev_time = curr_time

            # Interfaz visual
            draw_ui(frame, current_action, confidence, sentence, fps)

            cv.imshow('Reconocimiento LSCh Dinamico - TEL-328', frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                sentence.clear()
                print("[INFO] Historial limpiado.")

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
