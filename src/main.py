import argparse
import collections
import functools
import glob
import os
import sys
import time


# Auto-configurar rutas de bibliotecas NVIDIA si no están en LD_LIBRARY_PATH
def _ensure_cuda_libs():
    import ctypes
    site_packages = glob.glob(os.path.join(sys.prefix, 'lib', 'python*', 'site-packages', 'nvidia', '*', 'lib'))
    if not site_packages:
        return
    if sys.argv and sys.argv[0] not in ('-c', '') and not os.environ.get('_TF_CUDA_ENV_SET') and 'ipykernel' not in sys.modules:
        current_ld = os.environ.get('LD_LIBRARY_PATH', '')
        missing = [p for p in site_packages if p not in current_ld.split(':')]
        if missing:
            env = os.environ.copy()
            env['LD_LIBRARY_PATH'] = ':'.join(site_packages) + (':' + current_ld if current_ld else '')
            env['_TF_CUDA_ENV_SET'] = '1'
            os.execve(sys.executable, [sys.executable] + sys.argv, env)
    for d in site_packages:
        for so in sorted(glob.glob(os.path.join(d, '*.so.*'))):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass


_ensure_cuda_libs()

# Suprimir logs ruidosos de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

# Configuración y verificación de GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        details = tf.config.experimental.get_device_details(gpus[0])
        gpu_name = details.get('device_name', gpus[0].name)
        print(f"[GPU ACTIVA] TensorFlow ejecutando en GPU: {gpu_name}")
    except RuntimeError as e:
        print(f"[GPU AVISO] {e}")
else:
    print("[CPU] No se detectó GPU disponible, ejecutando en CPU.")

# Rutas del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'gru_lsch_dinamico.h5')
STATIC_MLP_PATH = os.path.join(BASE_DIR, '..', 'models', 'mlp_lsch.keras')
HOLISTIC_TASK_PATH = os.path.join(BASE_DIR, '..', 'models', 'holistic_landmarker.task')
HAND_TASK_PATH = os.path.join(BASE_DIR, '..', 'models', 'hand_landmarker.task')

# 50 Clases del modelo dinámico de Lengua de Señas Chilena (LSCh)
ACTIONS = np.array([
    'NONE',
    # Alfabeto LSCh estático (índices 1..21)
    'A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'P', 'R', 'T', 'U', 'V', 'W', 'Y', 'Q',
    # Alfabeto LSCh dinámico (índices 22..27)
    'G', 'J', 'Ñ', 'S', 'X', 'Z',
    # Palabras unimanuales LSCh (índices 28..36)
    'Hola', 'Chao', 'Comer', 'Caminar', 'Flaco', 'Alto', 'Bajo',
    'Bonito', 'Hablar',
    # Palabras bimanuales LSCh (índices 37..49)
    'Ayudar', 'Abrir', 'Disculpar', 'Vender', 'Dibujar', 'Planchar',
    'Amigo', 'Responsable', 'Arrendar', 'Chocar', 'Vestir', 'Estudiar', 'Jugar'
])

STATIC_MLP_LABELS = list('ABCDEFGHIKLMNOPQRSTUVWXY')
ACTION_TO_IDX = {act: i for i, act in enumerate(ACTIONS)}
PURE_STATIC_INDICES = list(range(1, 21))        # A..Y (sin Q)
STATIC_LETTER_INDICES = list(range(1, 22))      # A..Q
DYNAMIC_UNIMANUAL_INDICES = list(range(22, 37)) # G..Z + Hola..Hablar
UNIMANUAL_INDICES = list(range(1, 37))          # 1..36
BIMANUAL_INDICES = list(range(37, 50))          # 37..49

SEQUENCE_LENGTH = 30  # 30 frames temporales (~1 segundo)
NUM_FEATURES = 258    # Cinemática y geometría 3D (132) + Mano Izq (63) + Mano Der (63)
KEY_PTS = [0, 4, 8, 12, 16, 20]  # Muñeca + 5 puntas de dedos
SUB_PTS = [0, 4, 8, 20]          # Muñeca, pulgar, índice, meñique

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


def load_static_mlp(mlp_path):
    """Carga el modelo MLP auxiliar para reforzar letras estáticas."""
    if os.path.exists(mlp_path):
        try:
            mlp = tf.keras.models.load_model(mlp_path)
            print(f"[OK] Modelo estático auxiliar MLP cargado desde {mlp_path}")
            return mlp
        except Exception as e:
            print(f"[AVISO] No se pudo cargar MLP estático auxiliar: {e}")
    return None


class RobustHandTracker:
    """
    Rastreador temporal de manos de alta robustez:
    - Previene que las manos se confundan entre sí o se pierdan contra el torso.
    - Respeta incondicionalmente ambas manos cuando Holistic las detecta (no borra manos cuando se tocan).
    - Asigna candidatos de HandLandmarker por continuidad temporal y conectividad con los brazos (muñecas 15 y 16 de Pose).
    - Mantiene 'coasting' por oclusión en señas bimanuales de contacto (Chocar, Amigo, Estudiar, Disculpar)
      para que una mano no desaparezca cuando la otra pasa por delante.
    """
    def __init__(self):
        self.prev_rh = None  # Lista de landmarks de RH en frame anterior
        self.prev_lh = None  # Lista de landmarks de LH en frame anterior
        self.rh_lost_frames = 0
        self.lh_lost_frames = 0
        self.max_occlusion_coasting = 6

    def reset(self):
        self.prev_rh = None
        self.prev_lh = None
        self.rh_lost_frames = 0
        self.lh_lost_frames = 0

    def update(self, pose_lms, hol_rh, hol_lh, hand_detector, mp_image):
        # 1. Si Holistic detectó ambas manos, confiar en ambas plenamente (evita 'ganar dominio')
        if hol_rh and hol_lh:
            self.prev_rh = hol_rh
            self.prev_lh = hol_lh
            self.rh_lost_frames = 0
            self.lh_lost_frames = 0
            return hol_rh, hol_lh

        curr_rh = hol_rh
        curr_lh = hol_lh

        # 2. Si falta alguna mano en Holistic, consultar HandLandmarker (detector de cuadro completo)
        if (not curr_rh or not curr_lh) and hand_detector is not None:
            hand_res = hand_detector.detect(mp_image)
            if hand_res and hand_res.hand_landmarks:
                candidates = list(hand_res.hand_landmarks)

                # Filtrar duplicados: un candidato solo es duplicado si coincide en todos los 21 puntos (< 0.045)
                unique_cands = []
                for cand in candidates:
                    is_dup = False
                    for existing in [curr_rh, curr_lh]:
                        if existing is not None:
                            mean_dist = np.mean([
                                np.hypot(c.x - e.x, c.y - e.y)
                                for c, e in zip(cand, existing)
                            ])
                            if mean_dist < 0.045:
                                is_dup = True
                                break
                    if not is_dup:
                        unique_cands.append(cand)

                # Coordenadas de referencia de los brazos desde Pose
                # En imagen original sin espejo: muñeca derecha = 16, muñeca izquierda = 15
                pose_rw = (pose_lms[16].x, pose_lms[16].y) if (pose_lms and len(pose_lms) > 16) else (0.35, 0.70)
                pose_lw = (pose_lms[15].x, pose_lms[15].y) if (pose_lms and len(pose_lms) > 15) else (0.65, 0.70)

                for cand in unique_cands:
                    cw = (cand[0].x, cand[0].y)
                    # Costo de asociar a Mano Derecha
                    d_rw = np.hypot(cw[0] - pose_rw[0], cw[1] - pose_rw[1])
                    if self.prev_rh is not None:
                        d_rw = 0.45 * d_rw + 0.55 * np.hypot(cw[0] - self.prev_rh[0].x, cw[1] - self.prev_rh[0].y)

                    # Costo de asociar a Mano Izquierda
                    d_lw = np.hypot(cw[0] - pose_lw[0], cw[1] - pose_lw[1])
                    if self.prev_lh is not None:
                        d_lw = 0.45 * d_lw + 0.55 * np.hypot(cw[0] - self.prev_lh[0].x, cw[1] - self.prev_lh[0].y)

                    if curr_rh is None and (curr_lh is not None or d_rw <= d_lw):
                        curr_rh = cand
                    elif curr_lh is None and (curr_rh is not None or d_lw < d_rw):
                        curr_lh = cand

        # 3. Persistencia por oclusión de contacto (cuando las dos manos se tocan o chocan)
        if curr_rh is not None and curr_lh is None and self.prev_lh is not None:
            # Si la mano izquierda estaba activa recientemente y las manos estaban cerca, mantenerla viva
            d_contact = np.hypot(curr_rh[0].x - self.prev_lh[0].x, curr_rh[0].y - self.prev_lh[0].y)
            if d_contact < 0.16 and self.lh_lost_frames < self.max_occlusion_coasting:
                curr_lh = self.prev_lh
                self.lh_lost_frames += 1
            else:
                self.lh_lost_frames += 1
        elif curr_lh is not None:
            self.prev_lh = curr_lh
            self.lh_lost_frames = 0
        else:
            self.lh_lost_frames += 1

        if curr_lh is not None and curr_rh is None and self.prev_rh is not None:
            d_contact = np.hypot(curr_lh[0].x - self.prev_rh[0].x, curr_lh[0].y - self.prev_rh[0].y)
            if d_contact < 0.16 and self.rh_lost_frames < self.max_occlusion_coasting:
                curr_rh = self.prev_rh
                self.rh_lost_frames += 1
            else:
                self.rh_lost_frames += 1
        elif curr_rh is not None:
            self.prev_rh = curr_rh
            self.rh_lost_frames = 0
        else:
            self.rh_lost_frames += 1

        return curr_rh, curr_lh


def extract_keypoints(pose_lms, lh_lms, rh_lms):
    """Convierte los landmarks validados en el vector estándar de 258 características."""
    has_pose = bool(pose_lms and len(pose_lms) > 0)
    has_lh = bool(lh_lms and len(lh_lms) > 0)
    has_rh = bool(rh_lms and len(rh_lms) > 0)

    if has_pose:
        pose = np.array([
            [lm.x, lm.y, lm.z, getattr(lm, 'visibility', 0.0) or 0.0]
            for lm in pose_lms
        ], dtype=np.float32).flatten()
    else:
        pose = np.zeros(33 * 4, dtype=np.float32)

    if has_lh:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in lh_lms], dtype=np.float32).flatten()
    else:
        lh = np.zeros(21 * 3, dtype=np.float32)

    if has_rh:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in rh_lms], dtype=np.float32).flatten()
    else:
        rh = np.zeros(21 * 3, dtype=np.float32)

    features = np.concatenate([pose, lh, rh])
    return features, has_pose, has_rh, has_lh


def interpolate_dropped_hands(seq_raw):
    """Rellena caídas breves (hasta 5 frames) de detección de mano durante movimiento rápido."""
    seq = seq_raw.copy()
    T = len(seq)
    for start_col in [132, 195]:
        end_col = start_col + 63
        valid_mask = np.any(np.abs(seq[:, start_col:end_col]) > 1e-5, axis=1)
        if not np.any(valid_mask):
            continue
        last_valid = None
        gap = 0
        for t in range(T):
            if valid_mask[t]:
                last_valid = seq[t, start_col:end_col].copy()
                gap = 0
            else:
                gap += 1
                if last_valid is not None and gap <= 5:
                    seq[t, start_col:end_col] = last_valid
        first_valid_idx = int(np.argmax(valid_mask))
        if 0 < first_valid_idx <= 4:
            seq[:first_valid_idx, start_col:end_col] = seq[first_valid_idx, start_col:end_col]
    return seq


def unify_single_hand_sequence(seq_raw):
    """
    Si una seña es genuinamente unimanual (< 6 frames bimanuales), consolida la mano activa en RH
    (con reflejo en espejo si fue realizada con la mano izquierda por un usuario zurdo).
    Si es bimanual, preserva intactas ambas manos.
    """
    seq = seq_raw.copy()
    T = seq.shape[0]
    pose = seq[:, :132].reshape(T, 33, 4)
    lh = seq[:, 132:195].reshape(T, 21, 3)
    rh = seq[:, 195:258].reshape(T, 21, 3)

    has_lh = np.any(np.abs(lh) > 1e-5, axis=(1, 2))
    has_rh = np.any(np.abs(rh) > 1e-5, axis=(1, 2))
    both = has_lh & has_rh

    if np.sum(both) < 6:
        # Seña unimanual
        if np.sum(has_lh) > np.sum(has_rh) and np.sum(has_lh) >= 6:
            # Realizada predominantemente con la mano izquierda: reflejar en espejo hacia RH
            valid_p = np.linalg.norm(pose[:, :, :2], axis=2) > 1e-4
            pose[:, :, 0] = np.where(valid_p, 1.0 - pose[:, :, 0], 0.0)
            for a, b in [(11, 12), (13, 14), (15, 16)]:
                pose[:, [a, b], :] = pose[:, [b, a], :]
            new_rh = lh.copy()
            new_rh[has_lh, :, 0] = 1.0 - new_rh[has_lh, :, 0]
            rh = np.where(has_lh[:, None, None], new_rh, rh)
            lh[:] = 0.0
        elif np.sum(has_rh) >= 6:
            # Realizada con mano derecha: limpiar rastro incidental de la mano izquierda
            lh[:] = 0.0

    return np.concatenate([pose.reshape(T, 132), lh.reshape(T, 63), rh.reshape(T, 63)], axis=1)


def _compute_3d_hand_geom(h_norm, has_h):
    """Calcula 14 distancias 3D invariantes a rotación de cámara/muñeca."""
    d_ext = [np.linalg.norm(h_norm[:, j, :] - h_norm[:, 9, :], axis=1) for j in [4, 8, 12, 16, 20]]
    d_pinch = [np.linalg.norm(h_norm[:, 4, :] - h_norm[:, j, :], axis=1) for j in [8, 12, 16, 20]]
    d_adj = [
        np.linalg.norm(h_norm[:, 8, :] - h_norm[:, 5, :], axis=1),
        np.linalg.norm(h_norm[:, 8, :] - h_norm[:, 12, :], axis=1),
        np.linalg.norm(h_norm[:, 12, :] - h_norm[:, 16, :], axis=1),
        np.linalg.norm(h_norm[:, 16, :] - h_norm[:, 20, :], axis=1),
    ]
    d_opp = [np.linalg.norm(h_norm[:, 4, :] - h_norm[:, 17, :], axis=1)]
    geom = np.stack(d_ext + d_pinch + d_adj + d_opp, axis=1)  # (T, 14)
    return np.where(has_h[:, None], geom, 0.0)


def normalize_sequence(seq_raw):
    """
    Normaliza de forma vectorizada una secuencia de 30 frames (30, 258):
    - Trayectoria centrada explícita (invariante a la posición donde el usuario hace la seña)
    - Distancias 3D internas de los dedos (invariantes al ángulo 3D de la cámara/mano)
    - Auto-normalización de escala XY y profundidad Z
    - Inmune al peso/inclinación del torso
    """
    seq = unify_single_hand_sequence(interpolate_dropped_hands(np.asarray(seq_raw, dtype=np.float32)))
    T = seq.shape[0]

    pose = seq[:, :132].reshape(T, 33, 4)
    lh = seq[:, 132:195].reshape(T, 21, 3)
    rh = seq[:, 195:258].reshape(T, 21, 3)

    has_lh = np.any(np.abs(lh) > 1e-5, axis=(1, 2))
    has_rh = np.any(np.abs(rh) > 1e-5, axis=(1, 2))
    has_any_hand = has_lh | has_rh

    if not np.any(has_any_hand):
        return np.zeros((T, 258), dtype=np.float32)

    l_sh = pose[:, 11, :2]
    r_sh = pose[:, 12, :2]
    nose = pose[:, 0, :2]

    valid_sh = (np.linalg.norm(l_sh, axis=1) > 1e-3) & (np.linalg.norm(r_sh, axis=1) > 1e-3)
    neck = np.where(valid_sh[:, None], 0.5 * (l_sh + r_sh), np.array([0.5, 0.72], dtype=np.float32))
    sh_dist = np.linalg.norm(l_sh - r_sh, axis=1)
    body_scale = np.where(valid_sh, np.clip(sh_dist, 0.18, 0.70), 0.38).astype(np.float32)

    valid_nose = np.linalg.norm(nose, axis=1) > 1e-3
    default_head = neck - np.stack([np.zeros(T, dtype=np.float32), 0.65 * body_scale], axis=1)
    head = np.where(valid_nose[:, None], nose, default_head)

    # 1. Normalización local de Mano Izquierda (T, 21, 3)
    lh_c = lh - lh[:, 0:1, :]
    lh_span = np.max(np.linalg.norm(lh_c[:, :, :2], axis=2), axis=1)
    lh_scale_xy = np.maximum(lh_span, 0.04)[:, None]
    lh_scale_z = np.maximum(np.max(np.abs(lh_c[:, :, 2]), axis=1), 0.02)[:, None]
    lh_norm = np.zeros((T, 21, 3), dtype=np.float32)
    lh_norm[:, :, :2] = np.where(has_lh[:, None, None], lh_c[:, :, :2] / lh_scale_xy[:, :, None], 0.0)
    lh_norm[:, :, 2] = np.where(has_lh[:, None], (lh_c[:, :, 2] / lh_scale_z) * 0.25, 0.0)
    lh_norm[:, 0, 0] = np.where(has_lh, 1.0, 0.0)
    lh_norm[:, 0, 1] = np.where(has_lh, np.clip(lh_span / body_scale, 0.0, 2.0), 0.0)
    lh_norm[:, 0, 2] = 0.0

    # 2. Normalización local de Mano Derecha (T, 21, 3)
    rh_c = rh - rh[:, 0:1, :]
    rh_span = np.max(np.linalg.norm(rh_c[:, :, :2], axis=2), axis=1)
    rh_scale_xy = np.maximum(rh_span, 0.04)[:, None]
    rh_scale_z = np.maximum(np.max(np.abs(rh_c[:, :, 2]), axis=1), 0.02)[:, None]
    rh_norm = np.zeros((T, 21, 3), dtype=np.float32)
    rh_norm[:, :, :2] = np.where(has_rh[:, None, None], rh_c[:, :, :2] / rh_scale_xy[:, :, None], 0.0)
    rh_norm[:, :, 2] = np.where(has_rh[:, None], (rh_c[:, :, 2] / rh_scale_z) * 0.25, 0.0)
    rh_norm[:, 0, 0] = np.where(has_rh, 1.0, 0.0)
    rh_norm[:, 0, 1] = np.where(has_rh, np.clip(rh_span / body_scale, 0.0, 2.0), 0.0)
    rh_norm[:, 0, 2] = 0.0

    # 3. Vector cinemático y geométrico (T, 132)
    kin = np.zeros((T, 132), dtype=np.float32)

    kin[:, 0] = has_lh.astype(np.float32)
    kin[:, 1] = has_rh.astype(np.float32)
    kin[:, 2] = (has_lh & has_rh).astype(np.float32)
    kin[:, 3] = (has_rh & (~has_lh)).astype(np.float32)

    bs = body_scale[:, None, None]
    rh_k = rh[:, KEY_PTS, :2]
    lh_k = lh[:, KEY_PTS, :2]

    # [4..28] Trayectoria centrada en la media de la ventana
    if np.any(has_rh):
        rh_mean = np.mean(rh_k[has_rh], axis=0, keepdims=True)
        rh_traj = np.clip((rh_k - rh_mean) / bs * 2.0, -3.0, 3.0).reshape(T, 12)
        kin[:, 4:16] = np.where(has_rh[:, None], rh_traj, 0.0)
    if np.any(has_lh):
        lh_mean = np.mean(lh_k[has_lh], axis=0, keepdims=True)
        lh_traj = np.clip((lh_k - lh_mean) / bs * 2.0, -3.0, 3.0).reshape(T, 12)
        kin[:, 16:28] = np.where(has_lh[:, None], lh_traj, 0.0)

    # [28..44] Desplazamiento acumulado desde el inicio del gesto
    if np.any(has_rh):
        first_rh = rh[int(np.argmax(has_rh)), SUB_PTS, :2]
        rh_disp = np.clip((rh[:, SUB_PTS, :2] - first_rh[None, :, :]) / bs * 1.5, -3.0, 3.0).reshape(T, 8)
        kin[:, 28:36] = np.where(has_rh[:, None], rh_disp, 0.0)
    if np.any(has_lh):
        first_lh = lh[int(np.argmax(has_lh)), SUB_PTS, :2]
        lh_disp = np.clip((lh[:, SUB_PTS, :2] - first_lh[None, :, :]) / bs * 1.5, -3.0, 3.0).reshape(T, 8)
        kin[:, 36:44] = np.where(has_lh[:, None], lh_disp, 0.0)

    # [44..60] Ancla anatómica suave respecto a cuello y cabeza
    rh_anchor_pts = rh[:, [0, 8], :2]
    lh_anchor_pts = lh[:, [0, 8], :2]
    kin[:, 44:48] = np.where(has_rh[:, None], np.clip((rh_anchor_pts - neck[:, None, :]) / bs * 0.6, -2.0, 2.0).reshape(T, 4), 0.0)
    kin[:, 48:52] = np.where(has_rh[:, None], np.clip((rh_anchor_pts - head[:, None, :]) / bs * 0.6, -2.0, 2.0).reshape(T, 4), 0.0)
    kin[:, 52:56] = np.where(has_lh[:, None], np.clip((lh_anchor_pts - neck[:, None, :]) / bs * 0.6, -2.0, 2.0).reshape(T, 4), 0.0)
    kin[:, 56:60] = np.where(has_lh[:, None], np.clip((lh_anchor_pts - head[:, None, :]) / bs * 0.6, -2.0, 2.0).reshape(T, 4), 0.0)

    # [60..72] Relación bimanual directa (RH - LH)
    both = has_lh & has_rh
    bimanual = np.clip((rh_k - lh_k) / bs, -3.0, 3.0).reshape(T, 12)
    kin[:, 60:72] = np.where(both[:, None], bimanual, 0.0)

    # [72..100] Geometría 3D invariante a rotación de mano/cámara (28 dims)
    kin[:, 72:86] = _compute_3d_hand_geom(rh_norm, has_rh)
    kin[:, 86:100] = _compute_3d_hand_geom(lh_norm, has_lh)

    # [100..132] Velocidades cuadro a cuadro y movimiento local centrado de dedos
    if np.any(has_rh):
        rh_loc_k = rh_norm[:, KEY_PTS, :2]
        rh_loc_mean = np.mean(rh_loc_k[has_rh], axis=0, keepdims=True)
        kin[:, 112:124] = np.where(has_rh[:, None], np.clip((rh_loc_k - rh_loc_mean) * 3.0, -3.0, 3.0).reshape(T, 12), 0.0)

    if T > 1:
        valid_v_rh = has_rh[1:] & has_rh[:-1]
        v_rh_glob = np.clip((rh_k[1:] - rh_k[:-1]) / bs[1:] * 5.0, -3.0, 3.0).reshape(T - 1, 12)
        kin[1:, 100:112] = np.where(valid_v_rh[:, None], v_rh_glob, 0.0)

        valid_v_lh = has_lh[1:] & has_lh[:-1]
        v_lh_glob = np.clip((lh[1:, SUB_PTS, :2] - lh[:-1, SUB_PTS, :2]) / bs[1:] * 5.0, -3.0, 3.0).reshape(T - 1, 8)
        kin[1:, 124:132] = np.where(valid_v_lh[:, None], v_lh_glob, 0.0)

    kin = np.where(has_any_hand[:, None], kin, 0.0)
    return np.concatenate([kin, lh_norm.reshape(T, 63), rh_norm.reshape(T, 63)], axis=1).astype(np.float32)


def compute_refined_probabilities(gru_pred, seq_raw, norm_seq, static_mlp=None):
    """
    Aplica fine-tuning cinemático según el Diccionario Oficial de LSCh (UMCE):
    - Diferencia de trayectoria vertical: Alto (hacia arriba) vs Bajo (hacia abajo) vs Flaco (meñique abajo).
    - Cinemática bimanual: Abrir (separación), Chocar (colisión), Amigo (manos juntas), Estudiar/Planchar/Dibujar (mano base).
    - Fusión con MLP estático cuando la mano está quieta para máxima precisión en letras.
    Retorna: (probabilidades_refinadas, energia_movimiento, modo_cinematico)
    """
    p = gru_pred.copy()
    p[0] = 0.0  # Con manos activas en escena, evaluar qué seña es

    both_frames = int(np.sum(norm_seq[:, 2] > 0.5))
    lh_frames = int(np.sum(norm_seq[:, 0] > 0.5))
    rh_frames = int(np.sum(norm_seq[:, 1] > 0.5))
    is_bimanual = (both_frames >= 6) or (lh_frames >= 8 and rh_frames >= 8)

    pose = seq_raw[:, :132].reshape(30, 33, 4)
    sh_dist = np.linalg.norm(pose[:, 11, :2] - pose[:, 12, :2], axis=1)
    valid_sh = sh_dist > 1e-3
    bs = float(np.mean(sh_dist[valid_sh])) if np.any(valid_sh) else 0.38
    neck_y = float(np.mean(0.5 * (pose[valid_sh, 11, 1] + pose[valid_sh, 12, 1]))) if np.any(valid_sh) else 0.70

    lh = seq_raw[:, 132:195].reshape(30, 21, 3)
    rh = seq_raw[:, 195:258].reshape(30, 21, 3)
    has_rh = np.any(np.abs(rh) > 1e-5, axis=(1, 2))
    has_lh = np.any(np.abs(lh) > 1e-5, axis=(1, 2))

    motion_energy = 0.0
    kinematic_mode = "ESTÁTICO"

    if is_bimanual:
        kinematic_mode = "BIMANUAL"
        p[UNIMANUAL_INDICES] *= 0.01

        # Ajuste bimanual basado en el Diccionario LSCh:
        if np.sum(has_lh & has_rh) >= 4:
            b_dists = np.linalg.norm(lh[:, 0, :2] - rh[:, 0, :2], axis=1) / max(bs, 0.1)
            valid_b = has_lh & has_rh
            d_init = b_dists[valid_b][0]
            d_min = np.min(b_dists[valid_b])
            d_end = b_dists[valid_b][-1]

            lh_travel = np.sum(np.linalg.norm(lh[1:, 0, :2] - lh[:-1, 0, :2], axis=1)) / max(bs, 0.1)
            rh_travel = np.sum(np.linalg.norm(rh[1:, 0, :2] - rh[:-1, 0, :2], axis=1)) / max(bs, 0.1)
            motion_energy = float(np.clip((lh_travel + rh_travel) / 1.8, 0.15, 1.0))

            # Abrir (Diccionario LSCh: manos parten juntas y se abren hacia los lados)
            if (d_end - d_min) > 0.32:
                p[ACTION_TO_IDX['Abrir']] *= 2.5

            # Chocar (Diccionario LSCh: manos parten separadas y colisionan al centro)
            if (d_init - d_min) > 0.32 and d_min < 0.75:
                p[ACTION_TO_IDX['Chocar']] *= 2.5

            # Estudiar / Planchar / Dibujar (Diccionario LSCh: mano no dominante es base plana)
            if lh_travel < 0.40 and rh_travel > 0.60:
                p[ACTION_TO_IDX['Dibujar']] *= 1.8
                p[ACTION_TO_IDX['Planchar']] *= 1.8
                p[ACTION_TO_IDX['Estudiar']] *= 1.8

            # Amigo (Diccionario LSCh: manos entrelazadas o en contacto al centro)
            if d_end < 0.70 and d_min < 0.65:
                p[ACTION_TO_IDX['Amigo']] *= 2.0
    else:
        p[BIMANUAL_INDICES] *= 0.01

        sub_norm = norm_seq[3:28]
        has_rh_sub = sub_norm[:, 1] > 0.5
        if np.sum(has_rh_sub) >= 5:
            w_traj = sub_norm[has_rh_sub, 4:6] / 2.0
            wrist_rng = float(np.linalg.norm(np.ptp(w_traj, axis=0)))
            f_traj = sub_norm[has_rh_sub, 114:124].reshape(-1, 5, 2) / 3.0
            finger_rng = float(np.max(np.linalg.norm(np.ptp(f_traj, axis=0), axis=1)))
            motion_energy = float(np.clip(max(wrist_rng / 0.55, finger_rng / 0.65), 0.0, 1.0))

            # Mano quieta en el espacio -> Letra estática del abecedario
            if wrist_rng < 0.14 and finger_rng < 0.20:
                kinematic_mode = "ESTÁTICO"
                p[DYNAMIC_UNIMANUAL_INDICES] *= 0.02
                if static_mlp is not None:
                    rh_valid = rh[has_rh]
                    if len(rh_valid) >= 4:
                        c_hands = rh_valid - rh_valid[:, 0:1, :]
                        absmax = np.maximum(np.max(np.abs(c_hands), axis=(1, 2), keepdims=True), 1e-5)
                        mlp_in = (c_hands / absmax).reshape(len(rh_valid), 63)
                        mlp_out = np.mean(static_mlp(mlp_in, training=False).numpy(), axis=0)
                        mlp_full = np.zeros_like(p)
                        for m_idx, lbl in enumerate(STATIC_MLP_LABELS):
                            if lbl in ACTION_TO_IDX and lbl not in ('G', 'S', 'X'):
                                mlp_full[ACTION_TO_IDX[lbl]] = mlp_out[m_idx]
                        p = 0.75 * p + 0.25 * mlp_full
            elif wrist_rng > 0.26 or finger_rng > 0.34:
                kinematic_mode = "DINÁMICO"
                p[PURE_STATIC_INDICES] *= 0.05

                # Ajuste unimanual dinámico según el Diccionario LSCh:
                if np.sum(has_rh) >= 6:
                    first_idx = int(np.argmax(has_rh))
                    last_idx = len(has_rh) - 1 - int(np.argmax(has_rh[::-1]))
                    dy = float((rh[last_idx, 0, 1] - rh[first_idx, 0, 1]) / max(bs, 0.1))
                    finish_y = float((rh[last_idx, 0, 1] - neck_y) / max(bs, 0.1))

                    # Alto (Diccionario LSCh: mano se desplaza verticalmente HACIA ARRIBA)
                    if dy < -0.40 or finish_y < -0.80:
                        p[ACTION_TO_IDX['Alto']] *= 3.0
                        p[ACTION_TO_IDX['Bajo']] *= 0.05
                        p[ACTION_TO_IDX['Flaco']] *= 0.05
                    elif dy > +0.35:
                        # Flaco vs Bajo (Diccionario LSCh: Flaco usa dedo meñique extendido)
                        rh_c = rh[last_idx] - rh[last_idx, 0]
                        p_len = float(np.linalg.norm(rh_c[20, :2]))
                        i_len = float(np.linalg.norm(rh_c[8, :2]))
                        if p_len > 1.45 * max(i_len, 0.02):
                            p[ACTION_TO_IDX['Flaco']] *= 3.0
                            p[ACTION_TO_IDX['Alto']] *= 0.05
                            p[ACTION_TO_IDX['Bajo']] *= 0.10
                        else:
                            p[ACTION_TO_IDX['Bajo']] *= 3.0
                            p[ACTION_TO_IDX['Alto']] *= 0.05
                            p[ACTION_TO_IDX['Flaco']] *= 0.10
            else:
                kinematic_mode = "TRANSICIÓN"

    total = np.sum(p)
    if total > 1e-6:
        p /= total
    return p, motion_energy, kinematic_mode


# Caché de fuentes TrueType para renderizado UTF-8 nítido (ñ, tildes, símbolos)
_FONT_PATHS_BOLD = [
    '/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf',
    '/usr/share/fonts/google-droid-sans-fonts/DroidSans-Bold.ttf',
    '/usr/share/fonts/TTF/DejaVuSans-Bold.ttf',
]
_FONT_PATHS_REG = [
    '/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf',
    '/usr/share/fonts/google-droid-sans-fonts/DroidSans.ttf',
    '/usr/share/fonts/TTF/DejaVuSans.ttf',
]


@functools.lru_cache(maxsize=32)
def _load_font(size, bold=True):
    candidates = _FONT_PATHS_BOLD if bold else _FONT_PATHS_REG
    for path in candidates + _FONT_PATHS_BOLD:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


@functools.lru_cache(maxsize=512)
def _render_text_sprite(text, size, color_bgr, bold=True):
    font = _load_font(size, bold=bold)
    bbox = font.getbbox(text)
    w = max(1, bbox[2] - bbox[0] + 4)
    h = max(1, bbox[3] - bbox[1] + 4)
    img = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(img)
    draw.text((-bbox[0] + 2, -bbox[1] + 2), text, font=font, fill=255)
    alpha_u16 = np.array(img, dtype=np.uint16)[:, :, None]
    color_u16 = np.array(color_bgr, dtype=np.uint16).reshape(1, 1, 3)
    return alpha_u16, color_u16


def draw_text_utf8(frame, text, pos, size=16, color=(255, 255, 255), bold=True):
    """Dibuja texto anti-aliased con soporte completo de español (Ñ, tildes) en <0.05 ms usando caché."""
    if not text:
        return 0, 0
    x, y = int(pos[0]), int(pos[1])
    alpha_u16, color_u16 = _render_text_sprite(str(text), int(size), tuple(int(c) for c in color), bool(bold))
    sh, sw = alpha_u16.shape[:2]
    fh, fw = frame.shape[:2]

    if x >= fw or y >= fh or x + sw <= 0 or y + sh <= 0:
        return sw, sh

    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + sw), min(fh, y + sh)
    sx0, sy0 = x0 - x, y0 - y
    sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)

    roi = frame[y0:y1, x0:x1].astype(np.uint16)
    a = alpha_u16[sy0:sy1, sx0:sx1]
    frame[y0:y1, x0:x1] = ((roi * (255 - a) + color_u16 * a) >> 8).astype(np.uint8)
    return sw, sh


def draw_landmarks_mirrored(frame, pose_lms, lh_lms, rh_lms, rh_trail=None, lh_trail=None):
    """
    Dibuja sobre el frame ya espejado los esqueletos de manos, extremidades superiores,
    estelas cinemáticas de trayectoria y etiquetas legibles de lateralidad.
    """
    h, w = frame.shape[:2]

    # 1. Estelas de movimiento (trayectoria reciente de las muñecas)
    if rh_trail and len(rh_trail) >= 2:
        pts_t = list(rh_trail)
        for i in range(1, len(pts_t)):
            ratio = i / len(pts_t)
            color = (int(60 * ratio), int(255 * ratio), int(180 * ratio))
            thickness = max(1, int(4 * ratio))
            cv.line(frame, pts_t[i - 1], pts_t[i], color, thickness, cv.LINE_AA)

    if lh_trail and len(lh_trail) >= 2:
        pts_t = list(lh_trail)
        for i in range(1, len(pts_t)):
            ratio = i / len(pts_t)
            color = (int(255 * ratio), int(80 * ratio), int(220 * ratio))
            thickness = max(1, int(4 * ratio))
            cv.line(frame, pts_t[i - 1], pts_t[i], color, thickness, cv.LINE_AA)

    # 2. Pose superior (Hombros, codos, muñecas)
    if pose_lms and len(pose_lms) > 0:
        pts = [(int((1.0 - lm.x) * w), int(lm.y * h)) for lm in pose_lms]
        for a, b in POSE_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv.line(frame, pts[a], pts[b], (210, 140, 50), 2, cv.LINE_AA)
        for idx in [11, 12, 13, 14, 15, 16]:
            if idx < len(pts):
                cv.circle(frame, pts[idx], 5, (40, 40, 40), -1, cv.LINE_AA)
                cv.circle(frame, pts[idx], 3, (0, 190, 255), -1, cv.LINE_AA)

    # 3. Mano Derecha (Verde Esmeralda / Cyan)
    if rh_lms and len(rh_lms) > 0:
        pts = [(int((1.0 - lm.x) * w), int(lm.y * h)) for lm in rh_lms]
        for a, b in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv.line(frame, pts[a], pts[b], (110, 250, 40), 2, cv.LINE_AA)
        for idx, p in enumerate(pts):
            r = 4 if idx in KEY_PTS else 3
            cv.circle(frame, p, r + 1, (20, 40, 20), -1, cv.LINE_AA)
            cv.circle(frame, p, r, (0, 255, 255) if idx in KEY_PTS else (150, 255, 100), -1, cv.LINE_AA)
        wx, wy = pts[0]
        cv.rectangle(frame, (wx - 28, wy + 10), (wx + 28, wy + 30), (20, 45, 30), -1)
        cv.rectangle(frame, (wx - 28, wy + 10), (wx + 28, wy + 30), (80, 240, 120), 1)
        draw_text_utf8(frame, "DER", (wx - 16, wy + 12), size=13, color=(120, 255, 160), bold=True)

    # 4. Mano Izquierda (Magenta / Rosa Neón)
    if lh_lms and len(lh_lms) > 0:
        pts = [(int((1.0 - lm.x) * w), int(lm.y * h)) for lm in lh_lms]
        for a, b in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv.line(frame, pts[a], pts[b], (230, 70, 255), 2, cv.LINE_AA)
        for idx, p in enumerate(pts):
            r = 4 if idx in KEY_PTS else 3
            cv.circle(frame, p, r + 1, (40, 20, 40), -1, cv.LINE_AA)
            cv.circle(frame, p, r, (255, 240, 60) if idx in KEY_PTS else (255, 130, 240), -1, cv.LINE_AA)
        wx, wy = pts[0]
        cv.rectangle(frame, (wx - 26, wy + 10), (wx + 26, wy + 30), (45, 20, 45), -1)
        cv.rectangle(frame, (wx - 26, wy + 10), (wx + 26, wy + 30), (240, 90, 255), 1)
        draw_text_utf8(frame, "IZQ", (wx - 14, wy + 12), size=13, color=(250, 160, 255), bold=True)


def draw_ui(frame, top_preds, sentence, fps, has_pose, has_rh, has_lh,
            motion_energy, kinematic_mode, threshold, flash_frames=0):
    """
    Interfaz gráfica HUD de nivel presentación para el proyecto TEL-328:
    - Cabecera superior con indicadores de estado (Cuerpo, Lateralidad de Manos, Modo Cinemático, FPS).
    - Tarjeta lateral traslúcida con predicción principal, Top-3 barras de probabilidad y medidor de energía.
    - Barra inferior de subtítulos / oración acumulada con chips visuales.
    """
    h, w = frame.shape[:2]
    has_any = has_rh or has_lh
    top1_action, top1_conf = top_preds[0] if top_preds else ('NONE', 0.0)
    is_confirmed = has_any and top1_conf >= threshold and top1_action != 'NONE'

    # Capas traslúcidas oscuras (efecto glassmorphism)
    overlay = frame.copy()
    # Barra superior
    cv.rectangle(overlay, (0, 0), (w, 44), (18, 20, 28), -1)
    # Panel lateral izquierdo de diagnóstico y predicciones
    panel_w = min(310, int(w * 0.48))
    cv.rectangle(overlay, (12, 54), (12 + panel_w, 224), (20, 24, 34), -1)
    # Barra inferior de oración
    cv.rectangle(overlay, (0, h - 58), (w, h), (16, 18, 25), -1)
    cv.addWeighted(overlay, 0.80, frame, 0.20, 0, dst=frame)

    # Líneas de acento sutiles
    cv.line(frame, (0, 44), (w, 44), (70, 180, 240), 1)
    cv.line(frame, (0, h - 58), (w, h - 58), (60, 90, 130), 1)
    border_col = (80, 245, 150) if (is_confirmed or flash_frames > 0) else (65, 85, 115)
    cv.rectangle(frame, (12, 54), (12 + panel_w, 224), border_col, 1, cv.LINE_AA)

    # 1. Cabecera Superior
    draw_text_utf8(frame, "LSCh • RECONOCIMIENTO DINÁMICO", (16, 12), size=16, color=(240, 245, 255), bold=True)

    # Indicador de Pose
    pose_col = (90, 240, 130) if has_pose else (90, 90, 230)
    pose_lbl = "CUERPO: OK" if has_pose else "CUERPO: NO"
    draw_text_utf8(frame, pose_lbl, (w - 355, 14), size=13, color=pose_col, bold=True)

    # Indicador de Manos
    if has_rh and has_lh:
        hands_lbl, hands_col = "BIMANUAL", (60, 235, 255)
    elif has_rh:
        hands_lbl, hands_col = "MANO DER", (110, 250, 140)
    elif has_lh:
        hands_lbl, hands_col = "MANO IZQ", (245, 120, 255)
    else:
        hands_lbl, hands_col = "REPOSO", (150, 155, 165)
    draw_text_utf8(frame, f"MANOS: {hands_lbl}", (w - 250, 14), size=13, color=hands_col, bold=True)

    # FPS
    draw_text_utf8(frame, f"{fps:4.1f} FPS", (w - 82, 14), size=13, color=(190, 210, 230), bold=True)

    # 2. Panel de Predicción Activa
    if not has_any:
        status_title = "En espera de señas..."
        status_color = (165, 175, 190)
        sub_hint = "Levanta una o ambas manos frente a la cámara"
        draw_text_utf8(frame, status_title, (24, 66), size=20, color=status_color, bold=True)
        draw_text_utf8(frame, sub_hint, (24, 96), size=13, color=(135, 145, 160), bold=False)
    else:
        if is_confirmed:
            badge_txt = f"SEÑA: {top1_action}"
            badge_col = (90, 255, 155)
        else:
            badge_txt = f"Analizando: {top1_action}"
            badge_col = (120, 215, 255)
        draw_text_utf8(frame, badge_txt, (24, 62), size=22, color=badge_col, bold=True)

        # Barras Top-3
        preds_to_show = top_preds[:3] if len(top_preds) >= 3 else top_preds
        for idx, (act_name, prob) in enumerate(preds_to_show):
            row_y = 98 + idx * 24
            lbl_str = f"#{idx + 1} {act_name}"
            draw_text_utf8(frame, lbl_str, (24, row_y), size=13,
                           color=(240, 240, 245) if idx == 0 else (175, 185, 200),
                           bold=(idx == 0))

            bar_x = 128
            bar_w = panel_w - 178
            bar_h = 10
            cv.rectangle(frame, (bar_x, row_y + 4), (bar_x + bar_w, row_y + 4 + bar_h), (45, 52, 68), -1)
            fill_w = int(bar_w * max(0.0, min(1.0, float(prob))))
            if idx == 0:
                fill_col = (75, 235, 130) if prob >= threshold else (50, 185, 255)
            else:
                fill_col = (110, 130, 165)
            if fill_w > 0:
                cv.rectangle(frame, (bar_x, row_y + 4), (bar_x + fill_w, row_y + 4 + bar_h), fill_col, -1)

            pct_str = f"{int(round(prob * 100)):3d}%"
            draw_text_utf8(frame, pct_str, (bar_x + bar_w + 8, row_y), size=13,
                           color=(235, 240, 250) if idx == 0 else (165, 175, 190), bold=(idx == 0))

    # 3. Medidor de Energía Cinemática y Modo (Estático / Dinámico / Bimanual)
    mode_y = 180
    cv.line(frame, (22, mode_y - 6), (panel_w, mode_y - 6), (45, 55, 75), 1)
    mode_badge_col = (70, 230, 255) if kinematic_mode == "BIMANUAL" else (
        (90, 200, 255) if kinematic_mode == "DINÁMICO" else (160, 230, 160)
    )
    draw_text_utf8(frame, f"Modo: {kinematic_mode if has_any else 'REPOSO'}", (24, mode_y),
                   size=13, color=mode_badge_col if has_any else (140, 145, 155), bold=True)

    # Barra de energía de movimiento
    e_bar_x = 165
    e_bar_w = panel_w - 175
    draw_text_utf8(frame, "Cinemática", (24, mode_y + 20), size=12, color=(165, 175, 190), bold=False)
    cv.rectangle(frame, (e_bar_x, mode_y + 24), (e_bar_x + e_bar_w, mode_y + 32), (45, 52, 68), -1)
    e_fill = int(e_bar_w * max(0.0, min(1.0, float(motion_energy if has_any else 0.0))))
    if e_fill > 0:
        e_col = (60, 190, 255) if motion_energy > 0.30 else (120, 235, 150)
        cv.rectangle(frame, (e_bar_x, mode_y + 24), (e_bar_x + e_fill, mode_y + 32), e_col, -1)

    # 4. Barra Inferior: Oración Acumulada y Controles
    draw_text_utf8(frame, "ORACIÓN:", (16, h - 44), size=14, color=(110, 205, 255), bold=True)
    if not sentence:
        draw_text_utf8(frame, "Realiza señas de forma continua para construir una oración...",
                       (102, h - 44), size=14, color=(135, 145, 165), bold=False)
    else:
        chip_x = 102
        visible_words = sentence[-7:]
        for w_idx, word in enumerate(visible_words):
            is_last = (w_idx == len(visible_words) - 1)
            approx_w = max(34, len(word) * 10 + 18)
            if chip_x + approx_w > w - 16:
                break
            chip_bg = (35, 85, 60) if is_last else (38, 46, 64)
            chip_border = (90, 240, 145) if is_last else (80, 100, 135)
            cv.rectangle(frame, (chip_x, h - 48), (chip_x + approx_w, h - 22), chip_bg, -1)
            cv.rectangle(frame, (chip_x, h - 48), (chip_x + approx_w, h - 22), chip_border, 1, cv.LINE_AA)
            draw_text_utf8(frame, word, (chip_x + 9, h - 44), size=15,
                           color=(255, 255, 255) if is_last else (215, 225, 240), bold=True)
            chip_x += approx_w + 8

    draw_text_utf8(frame, "Controles:  [Q] Salir   |   [C] Limpiar oración   |   [Retroceso] Borrar última seña",
                   (16, h - 18), size=12, color=(150, 160, 180), bold=False)


def open_webcam(preferred_idx=0):
    """Busca y abre la cámara especificada o la primera disponible."""
    candidates = [preferred_idx, 0, 1, 2]
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
    parser = argparse.ArgumentParser(description="Reconocimiento dinámico continuo de LSCh en tiempo real")
    parser.add_argument('--camera', type=int, default=0, help="Índice preferido de cámara (default: 0)")
    parser.add_argument('--threshold', type=float, default=0.65, help="Umbral mínimo de confianza (default: 0.65)")
    args = parser.parse_args()

    model = load_dynamic_model(MODEL_PATH)
    static_mlp = load_static_mlp(STATIC_MLP_PATH)

    if not os.path.exists(HOLISTIC_TASK_PATH):
        raise FileNotFoundError(f"No se encontró el modelo de MediaPipe en: {HOLISTIC_TASK_PATH}")

    holistic_options = vision.HolisticLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HOLISTIC_TASK_PATH),
        running_mode=vision.RunningMode.VIDEO
    )

    hand_options = None
    if os.path.exists(HAND_TASK_PATH):
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=HAND_TASK_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.30,  # Alta sensibilidad para no perder manos contra el torso
            min_hand_presence_confidence=0.30
        )

    cap, active_cam = open_webcam(args.camera)
    if cap is None:
        print("[ERROR] No se pudo acceder a ninguna cámara disponible.")
        return

    tracker = RobustHandTracker()
    sequence = collections.deque(maxlen=SEQUENCE_LENGTH)
    hand_flags = collections.deque(maxlen=SEQUENCE_LENGTH)
    prob_history = collections.deque(maxlen=4)
    rh_trail = collections.deque(maxlen=14)
    lh_trail = collections.deque(maxlen=14)

    sentence = []
    top_preds = [('NONE', 0.0)]
    motion_energy = 0.0
    kinematic_mode = "REPOSO"

    confirm_count = 0
    REQUIRED_CONSECUTIVE_FRAMES = 3
    MIN_ACTIVE_HAND_FRAMES = 10
    MAX_IDLE_FRAMES_BEFORE_RESET = 5
    SAME_SIGN_COOLDOWN_FRAMES = 38  # Permite repetir una misma seña en flujo continuo tras ~1.2s
    idle_frames = 0
    last_confirmed_action = 'NONE'
    frames_since_last_commit = 999
    flash_frames = 0

    prev_time = time.time()
    start_time_ms = int(time.monotonic() * 1000)
    last_ts_ms = -1
    fps = 30.0

    print("\n" + "=" * 76)
    print("  RECONOCIMIENTO DINÁMICO CONTINUO LSCh - PROYECTO PDI (TEL-328)")
    print("  - Clases activas: 50 (Abecedario estático/dinámico + Vocabulario LSCh)")
    print("  - Umbral de confianza:", args.threshold)
    print("  - Rastreo bimanual persistente + Fine-tuning Diccionario Oficial UMCE")
    print("  - Controles: [q] Salir | [c] Limpiar oración | [Backspace] Borrar última")
    print("=" * 76 + "\n")

    holistic_ctx = vision.HolisticLandmarker.create_from_options(holistic_options)
    hand_ctx = vision.HandLandmarker.create_from_options(hand_options) if hand_options else None

    with holistic_ctx as landmarker:
        try:
            while cap.isOpened():
                ret, raw_frame = cap.read()
                if not ret:
                    print("[AVISO] Flujo de video finalizado.")
                    break

                now_ms = int(time.monotonic() * 1000) - start_time_ms
                if now_ms <= last_ts_ms:
                    now_ms = last_ts_ms + 1
                last_ts_ms = now_ms

                raw_rgb = cv.cvtColor(raw_frame, cv.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=raw_rgb)

                holistic_result = landmarker.detect_for_video(mp_image, now_ms)
                pose_lms = holistic_result.pose_landmarks if (holistic_result and holistic_result.pose_landmarks) else None
                hol_rh = holistic_result.right_hand_landmarks if (holistic_result and holistic_result.right_hand_landmarks) else None
                hol_lh = holistic_result.left_hand_landmarks if (holistic_result and holistic_result.left_hand_landmarks) else None

                # Actualizar manos a través del RobustHandTracker
                tracked_rh, tracked_lh = tracker.update(pose_lms, hol_rh, hol_lh, hand_ctx, mp_image)

                keypoints, has_pose, has_rh, has_lh = extract_keypoints(pose_lms, tracked_lh, tracked_rh)
                has_any_hand = has_rh or has_lh
                frames_since_last_commit += 1
                if flash_frames > 0:
                    flash_frames -= 1

                h_f, w_f = raw_frame.shape[:2]
                if has_rh and tracked_rh:
                    rh_trail.append((int((1.0 - tracked_rh[0].x) * w_f), int(tracked_rh[0].y * h_f)))
                else:
                    rh_trail.clear()

                if has_lh and tracked_lh:
                    lh_trail.append((int((1.0 - tracked_lh[0].x) * w_f), int(tracked_lh[0].y * h_f)))
                else:
                    lh_trail.clear()

                if has_any_hand:
                    idle_frames = 0
                    sequence.append(keypoints)
                    hand_flags.append(1)
                else:
                    idle_frames += 1
                    if idle_frames >= MAX_IDLE_FRAMES_BEFORE_RESET:
                        sequence.clear()
                        hand_flags.clear()
                        tracker.reset()
                        prob_history.clear()
                        rh_trail.clear()
                        lh_trail.clear()
                        confirm_count = 0
                        last_confirmed_action = 'NONE'
                        top_preds = [('NONE', 1.0)]
                        motion_energy = 0.0
                        kinematic_mode = "REPOSO"
                    else:
                        sequence.append(keypoints)
                        hand_flags.append(0)

                # Espejar frame primero y luego dibujar esqueletos y etiquetas legibles
                frame = cv.flip(raw_frame, 1)
                draw_landmarks_mirrored(frame, pose_lms, tracked_lh, tracked_rh, rh_trail=rh_trail, lh_trail=lh_trail)

                active_count = sum(hand_flags)

                if has_any_hand and active_count >= MIN_ACTIVE_HAND_FRAMES and len(sequence) >= 14:
                    seq_arr = np.array(sequence, dtype=np.float32)
                    if len(seq_arr) < SEQUENCE_LENGTH:
                        idx = np.clip(np.round(np.linspace(0, len(seq_arr) - 1, SEQUENCE_LENGTH)).astype(int), 0, len(seq_arr) - 1)
                        seq_arr = seq_arr[idx]

                    norm_seq = normalize_sequence(seq_arr)
                    input_data = np.expand_dims(norm_seq, axis=0)
                    raw_pred = model(input_data, training=False).numpy()[0]

                    refined_pred, motion_energy, kinematic_mode = compute_refined_probabilities(
                        raw_pred, seq_arr, norm_seq, static_mlp=static_mlp
                    )
                    prob_history.append(refined_pred)
                    smooth_pred = np.mean(prob_history, axis=0)

                    top_indices = np.argsort(smooth_pred)[::-1][:3]
                    top_preds = [(ACTIONS[i], float(smooth_pred[i])) for i in top_indices]
                    top1_action, top1_conf = top_preds[0]

                    if top1_conf >= args.threshold and top1_action != 'NONE':
                        if top1_action == last_confirmed_action:
                            confirm_count += 1
                        else:
                            confirm_count = 1
                            last_confirmed_action = top1_action

                        if confirm_count >= REQUIRED_CONSECUTIVE_FRAMES:
                            can_commit = (
                                len(sentence) == 0
                                or sentence[-1] != top1_action
                                or frames_since_last_commit >= SAME_SIGN_COOLDOWN_FRAMES
                            )
                            if can_commit:
                                sentence.append(top1_action)
                                frames_since_last_commit = 0
                                flash_frames = 10
                                print(f"[RECONOCIDO]: {top1_action} ({int(round(top1_conf * 100))}%) [{kinematic_mode}]")

                                # Transición continua limpia hacia la siguiente seña sin obligar a bajar las manos
                                keep_frames = 8 if ACTION_TO_IDX.get(top1_action, 0) >= 22 else 14
                                while len(sequence) > keep_frames:
                                    sequence.popleft()
                                    hand_flags.popleft()
                                prob_history.clear()
                                confirm_count = 0
                    else:
                        confirm_count = 0

                curr_time = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
                prev_time = curr_time

                draw_ui(
                    frame, top_preds, sentence, fps, has_pose, has_rh, has_lh,
                    motion_energy, kinematic_mode, args.threshold, flash_frames=flash_frames
                )

                cv.imshow('Reconocimiento Dinamico LSCh - TEL-328', frame)
                key = cv.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('c'):
                    sentence.clear()
                    frames_since_last_commit = 999
                    print("[INFO] Oración limpiada.")
                elif key in (8, 127):  # Backspace / Delete
                    if sentence:
                        removed = sentence.pop()
                        frames_since_last_commit = 999
                        print(f"[INFO] Última seña eliminada: {removed}")
        finally:
            if hand_ctx is not None:
                hand_ctx.close()

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
