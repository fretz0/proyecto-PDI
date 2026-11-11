#!/usr/bin/env python3
"""
Script de verificación de entorno y aceleración por GPU (NVIDIA CUDA / cuDNN)
para el proyecto Reconocimiento de Lengua de Señas Chilena (LSCh) - PDI.
"""

import glob
import os
import sys

# Auto-configurar bibliotecas NVIDIA si no están en LD_LIBRARY_PATH
def _ensure_cuda_libs():
    site_packages = glob.glob(os.path.join(sys.prefix, 'lib', 'python*', 'site-packages', 'nvidia', '*', 'lib'))
    if site_packages and not os.environ.get('_TF_CUDA_ENV_SET'):
        current_ld = os.environ.get('LD_LIBRARY_PATH', '')
        missing = [p for p in site_packages if p not in current_ld.split(':')]
        if missing:
            env = os.environ.copy()
            env['LD_LIBRARY_PATH'] = ':'.join(site_packages) + (':' + current_ld if current_ld else '')
            env['_TF_CUDA_ENV_SET'] = '1'
            os.execve(sys.executable, [sys.executable] + sys.argv, env)

_ensure_cuda_libs()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

def main():
    print("=" * 65)
    print("  VERIFICACIÓN DE DEPENDENCIAS Y ACELERACIÓN POR GPU (CUDA)")
    print("=" * 65)
    
    print(f"\n[1] Versión de Python: {sys.version.split()[0]}")
    print(f"    Ruta ejecutable:   {sys.executable}")
    print(f"    Versión TensorFlow: {tf.__version__}")
    
    # 2. Detección de dispositivos
    print("\n[2] Comprobación de Dispositivos:")
    physical_devices = tf.config.list_physical_devices()
    for dev in physical_devices:
        print(f"    - {dev.device_type}: {dev.name}")
        
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("\n[FALLO] No se detectó ninguna GPU para TensorFlow.")
        print("Asegúrate de ejecutar este script dentro del entorno virtual .venv:")
        print("  source .venv/bin/activate")
        print("  python verify_gpu.py")
        sys.exit(1)
        
    for i, gpu in enumerate(gpus):
        tf.config.experimental.set_memory_growth(gpu, True)
        details = tf.config.experimental.get_device_details(gpu)
        print(f"\n[OK] GPU #{i} Detectada:")
        print(f"     Nombre:              {details.get('device_name', 'Desconocido')}")
        print(f"     Capacidad de cómputo: {details.get('compute_capability', 'N/A')}")
        
    # 3. Test de cómputo en GPU
    print("\n[3] Prueba de Cómputo Matricial en GPU...")
    with tf.device('/GPU:0'):
        x = tf.random.normal([2000, 2000])
        y = tf.matmul(x, x)
        print(f"    Multiplicación matricial (2000x2000) completada con éxito.")
        print(f"    Tensor colocado en: {y.device}")

    # 4. Verificación de carga e inferencia de modelos del proyecto
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mlp_path = os.path.join(base_dir, 'models', 'mlp_lsch.keras')
    gru_path = os.path.join(base_dir, 'models', 'gru_lsch_dinamico.h5')
    
    print("\n[4] Prueba de Inferencia con los Modelos del Proyecto:")
    if os.path.exists(mlp_path):
        mlp = tf.keras.models.load_model(mlp_path)
        dummy_mlp = np.zeros((1, 63), dtype=np.float32)
        out_mlp = mlp(dummy_mlp, training=False)
        print(f"    [OK] Modelo MLP ({os.path.basename(mlp_path)}): inferencia ejecutada en {out_mlp.device}")
    else:
        print(f"    [WARN] No se encontró {mlp_path}")

    if os.path.exists(gru_path):
        from tensorflow.keras.layers import Dense, GRU, Input
        from tensorflow.keras.models import Sequential
        gru = Sequential([
            Input(shape=(30, 258)),
            GRU(64, return_sequences=True, activation='tanh'),
            GRU(64, return_sequences=False, activation='tanh'),
            Dense(50, activation='softmax')
        ])
        gru.load_weights(gru_path)
        dummy_gru = np.zeros((1, 30, 258), dtype=np.float32)
        out_gru = gru(dummy_gru, training=False)
        print(f"    [OK] Modelo GRU ({os.path.basename(gru_path)}): inferencia ejecutada en {out_gru.device}")
    else:
        print(f"    [WARN] No se encontró {gru_path}")

    # 5. MediaPipe y OpenCV
    print("\n[5] Verificación de Paquetes Auxiliares:")
    import cv2
    import mediapipe
    print(f"    - OpenCV version:    {cv2.__version__}")
    print(f"    - MediaPipe version: {mediapipe.__version__}")

    print("\n" + "=" * 65)
    print("  ¡TODO CORRECTO! EL PROYECTO ESTÁ LISTO Y EJECUTÁNDOSE EN GPU.")
    print("=" * 65 + "\n")

if __name__ == '__main__':
    main()
