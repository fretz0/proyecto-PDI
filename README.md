# Reconocimiento Continuo de Lengua de Señas Chilena (LSCh) - PDI (TEL-328)

Sistema híbrido de visión por computador y aprendizaje profundo para el reconocimiento en **tiempo continuo** de la Lengua de Señas Chilena (LSCh). Soporta **50 clases en vivo**, integrando el **alfabeto dactilológico completo (estático y con trayectoria)**, **vocabulario cotidiano unimanual y bimanual**, normalización cinemática 3D invariante al peso del torso y ajuste biomecánico basado en el **Diccionario Oficial de la LSCh (UMCE)**.

## Video de demostración

- [Demostración del sistema en Google Drive](https://drive.google.com/file/d/1eVWueIFO9i3UL1kuoAqHCHu4M0GUljVG/view?usp=sharing)

---

## Estructura del repositorio

```text
.
├── data/
│   ├── Senas de LSCh.v5i.tensorflow/                     # Imágenes y anotaciones del dataset estático (train, valid, test)
│   └── lsch_coordenadas_estaticas.csv                    # Coordenadas normalizadas estáticas (63 features + label + split)
├── models/
│   ├── gru_lsch_dinamico.h5                              # Modelo GRU con normalización cinemática 3D (50 clases LSCh)
│   ├── mlp_lsch.keras                                    # Modelo MLP auxiliar para clasificación de letras estáticas
│   ├── holistic_landmarker.task                          # Modelo MediaPipe Holistic (postura + ambas manos)
│   └── hand_landmarker.task                              # Modelo MediaPipe Hand Landmarker (respaldo bimanual de alta sensibilidad)
├── notebooks/
│   ├── Analisis_Estadistico_y_Tendencias_LSCh.ipynb      # Notebook auto-explicativo: estadísticas, cinemática 3D, tendencias y métricas
│   └── Preprocesamiento_Entrenamiento.ipynb              # Pipeline de extracción estática, entrenamiento MLP y evaluación
├── src/
│   ├── main.py                                           # Reconocimiento continuo en tiempo real (50 clases + interfaz HUD)
│   └── main_estatico.py                                  # Reconocimiento estático tradicional (24 letras con MLP)
├── Diccionario_LSCh_A-H.pdf                              # Referencia oficial léxica y cinemática de la LSCh (UMCE)
├── requirements.txt                                      # Dependencias del proyecto con soporte GPU (CUDA/cuDNN)
├── verify_gpu.py                                         # Script de diagnóstico y benchmark de ejecución en GPU
└── README.md
```

---

## Características técnicas principales

1. **Inmunidad al Peso del Torso y Normalización Cinemática 3D (`normalize_sequence`)**:
   - Elimina la dominancia energética del cuerpo (que representaba el 96.8% de la energía $L_2$ en coordenadas crudas) centrando las trayectorias en la ventana activa, escalando por el ancho biacromial (hombros) e incorporando **28 distancias euclidianas 3D intrínsecas** de los dedos, logrando invarianza a traslación, escala y rotación de cámara ($\pm 22^\circ$).
2. **Seguimiento Bimanual Persistente (`RobustHandTracker`)**:
   - Previene que las manos se pierdan contra el torso o se absorban entre sí cuando colisionan o se tocan (*Chocar*, *Amigo*, *Estudiar*, *Disculpar*), combinando validación de 21 articulaciones, asignación anatómico-temporal con las muñecas de `Pose` y persistencia por oclusión (*coasting*).
3. **Fine-Tuning Cinemático según Diccionario Oficial LSCh (UMCE)**:
   - Discrimina trayectorias verticales (*Alto* vs. *Bajo* vs. *Flaco*), cinemática bimanual relativa (*Abrir*, *Chocar*, *Amigo*, *Estudiar/Planchar/Dibujar*) y fusiona probabilidades con el clasificador estático `mlp_lsch.keras` cuando la mano permanece estacionaria.
4. **Interfaz HUD de Presentación en Tiempo Continuo**:
   - Renderizado tipográfico UTF-8 anti-aliased (`Ñ`, tildes), estelas visuales de trayectoria en ambas muñecas, medidor de energía cinemática en vivo (`ESTÁTICO`, `DINÁMICO`, `BIMANUAL`), barras de probabilidad **Top-3** y construcción fluida de oraciones sin necesidad de bajar las manos entre señas.

---

## Instalación y Configuración (Linux / Windows)

### Linux (Fedora / Ubuntu con GPU NVIDIA)

1. Crear y activar el entorno virtual:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

2. Instalar las dependencias (incluye `tensorflow[and-cuda]` para aceleración por GPU):
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Verificar la detección y aceleración por GPU:
   ```bash
   python verify_gpu.py
   ```

### Windows 10 / 11

TensorFlow 2.11 y versiones posteriores no soportan CUDA de forma nativa en Windows. Para evitar la caída de FPS en modo eager de TensorFlow, el proyecto detecta el sistema operativo y ejecuta los modelos GRU y MLP sobre PyTorch (CUDA si la GPU está activa o motor C++ en CPU) con los mismos pesos entrenados. En Windows también se habilita captura de cámara por DirectShow (`CAP_DSHOW`) y fuentes del sistema (`Segoe UI`, `Arial`, `Calibri`) para el HUD.

1. Crear y activar el entorno virtual (PowerShell):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   *(En CMD: `.venv\Scripts\activate.bat`)*

2. Instalar las dependencias:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Verificar entorno, hardware NVIDIA y latencia de inferencia:
   ```powershell
   python verify_gpu.py
   ```

---



# Windows
.venv\Scripts\jupyter notebook notebooks\
```
