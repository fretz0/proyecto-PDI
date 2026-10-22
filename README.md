# Reconocimiento de Lengua de Señas Chilena (LSCh) - PDI

Sistema de visión por computador para la detección de manos, extracción y normalización de coordenadas de landmarks, y clasificación en tiempo real de señas estáticas del alfabeto de la Lengua de Señas Chilena (LSCh).

## Video de demostración

- [Demostración del sistema en Google Drive](https://drive.google.com/file/d/1eVWueIFO9i3UL1kuoAqHCHu4M0GUljVG/view?usp=sharing)

## Estructura del repositorio

```text
.
├── data/
│   ├── Senas de LSCh.v5i.tensorflow/     # Imágenes y anotaciones del dataset (train, valid, test)
│   └── lsch_coordenadas_estaticas.csv    # Coordenadas normalizadas generadas (63 features + label + split)
├── models/
│   ├── hand_landmarker.task              # Archivo de modelo Hand Landmarker de MediaPipe
│   └── mlp_lsch.keras                    # Modelo MLP entrenado para clasificación de señas LSCh
├── notebooks/
│   └── Preprocesamiento_Entrenamiento.ipynb # Pipeline de extracción, normalización, entrenamiento y evaluación
├── src/
│   └── main.py                           # Captura en tiempo real, extracción de landmarks y clasificación con MLP
└── README.md
```

## Requisitos

- Python 3.10 o 3.11 (recomendado para compatibilidad con MediaPipe y TensorFlow).
- Cámara web (para la captura y predicción en tiempo real).

### Dependencias principales

- `opencv-python`
- `mediapipe`
- `numpy`
- `pandas`
- `tensorflow`
- `matplotlib`
- `scikit-learn`

## Instalación

1. Clonar el repositorio:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd proyecto-PDI
   ```

2. Crear y activar un entorno virtual:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # En Linux/macOS
   # .venv\Scripts\activate   # En Windows
   ```

3. Instalar las dependencias necesarias:
   ```bash
   pip install opencv-python mediapipe numpy pandas tensorflow matplotlib scikit-learn
   ```

4. Verificar los modelos en la carpeta `models/`:
   - `models/hand_landmarker.task`: Modelo de MediaPipe Solutions para detección de landmarks de manos. Si no existe, debe descargarse desde la documentación oficial de MediaPipe.
   - `models/mlp_lsch.keras`: Modelo MLP entrenado para clasificación de letras. Si no está presente, se genera ejecutando el notebook de entrenamiento.

## Uso y ejecución

### 1. Inferencia y visualización en tiempo real (`src/main.py`)

Inicia la cámara web y detecta los 21 landmarks de la mano con MediaPipe. Dibuja el grafo esquelético sobre el video, despliega un panel lateral con las coordenadas numéricas normalizadas y clasifica la postura manual en tiempo real con el modelo MLP (`models/mlp_lsch.keras`), imprimiendo la letra predicha en consola (24 clases estáticas, de la A a la Y excluyendo J y Z).

```bash
python src/main.py
```

- **Configuración de cámara**: Por defecto, `main.py` utiliza `cv.VideoCapture(1)`. Si la cámara principal no abre o se usa la integrada de una laptop, editar la línea 153 en `src/main.py` cambiando el índice a `0`:
  ```python
  cap = cv.VideoCapture(0)
  ```
- **Salir**: Presionar la tecla `q` en la ventana de video para finalizar la ejecución.

### 2. Preprocesamiento, entrenamiento y evaluación (`notebooks/Preprocesamiento_Entrenamiento.ipynb`)

El notebook contiene el flujo de trabajo sobre el dataset estático de señas:

1. **Lectura de anotaciones**: Carga las imágenes y bounding boxes desde `data/Senas de LSCh.v5i.tensorflow/`.
2. **Filtrado**: Excluye letras que requieren movimiento dinámico (`J` y `Z`).
3. **Extracción y normalización**:
   - Pasa cada muestra por el detector `HandLandmarker`. Si la detección directa falla, realiza un recorte con margen de 20 píxeles sobre el bounding box anotado.
   - Centra las coordenadas restando la posición de la muñeca `(0, 0, 0)` y escala dividiendo por el valor absoluto máximo de las coordenadas.
4. **Exportación tabular**: Guarda la matriz normalizada de 65 columnas (63 correspondientes a los 21 nodos × 3 ejes, más etiqueta y split) en `data/lsch_coordenadas_estaticas.csv`.
5. **Entrenamiento del clasificador (MLP)**:
   - Configura una red neuronal secuencial con capas densas (128 y 64 neuronas con activación ReLU, regularización por Dropout de 0.3 y 0.2) y capa de salida de 24 neuronas con Softmax.
   - Aplica ponderación de clases (`compute_class_weight`) para mitigar el desbalance en el dataset.
   - Entrena durante 80 épocas con optimizador Adam y pérdida `sparse_categorical_crossentropy`.
6. **Evaluación y exportación**:
   - Evalúa el rendimiento sobre el subconjunto de prueba (`test`).
   - Genera la matriz de confusión con `ConfusionMatrixDisplay` para analizar clasificaciones erróneas entre señas con configuraciones de mano similares (por ejemplo, M y N).
   - Guarda el modelo resultante en `models/mlp_lsch.keras`.

Para ejecutarlo:
```bash
jupyter notebook notebooks/Preprocesamiento_Entrenamiento.ipynb
```
