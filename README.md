# Reconocimiento de Lengua de Señas Chilena (LSCh) - PDI

Sistema de visión por computador para la detección de manos y extracción de coordenadas de landmarks en tiempo real, enfocado en el reconocimiento de señas estáticas del alfabeto de la Lengua de Señas Chilena (LSCh).

## Estructura del repositorio

```text
.
├── data/
│   ├── Senas de LSCh.v5i.tensorflow/     # Imágenes y anotaciones del dataset (train, valid, test)
│   └── lsch_coordenadas_estaticas.csv    # Coordenadas normalizadas generadas (63 features + label + split)
├── models/
│   └── hand_landmarker.task              # Archivo de modelo Hand Landmarker de MediaPipe
├── notebooks/
│   └── Preprocesamiento_Entrenamiento.ipynb # Extracción de coordenadas, normalización y entrenamiento
├── src/
│   └── main.py                           # Captura de video en tiempo real y visualización del grafo
└── README.md
```

## Requisitos

- Python 3.10 o 3.11 (recomendado para compatibilidad con MediaPipe y TensorFlow).
- Cámara web (para la captura en tiempo real).

### Dependencias principales

- `opencv-python`
- `mediapipe`
- `numpy`
- `pandas`
- `tensorflow`
- `matplotlib`

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
   pip install opencv-python mediapipe numpy pandas tensorflow matplotlib
   ```

4. Verificar que el archivo `models/hand_landmarker.task` esté presente en la raíz del proyecto. Si no existe, debe descargarse desde la documentación oficial de MediaPipe Solutions.

## Uso y ejecución

### 1. Visualización en tiempo real (`src/main.py`)

Inicia la cámara web, detecta los 21 landmarks de la mano, dibuja el grafo esquelético y despliega un panel lateral con los valores numéricos `(x, y, z)` de cada punto clave.

```bash
python src/main.py
```

- **Configuración de cámara**: Por defecto, `main.py` utiliza `cv.VideoCapture(1)`. Si la cámara principal no abre o se usa la integrada de una laptop, editar la línea 135 en `src/main.py` cambiando el índice a `0`:
  ```python
  cap = cv.VideoCapture(0)
  ```
- **Salir**: Presionar la tecla `q` en la ventana de video para finalizar la ejecución.

### 2. Preprocesamiento y entrenamiento (`notebooks/Preprocesamiento_Entrenamiento.ipynb`)

El notebook contiene el flujo de trabajo sobre el dataset estático de señas:

1. **Lectura de anotaciones**: Carga las imágenes y bounding boxes desde `data/Senas de LSCh.v5i.tensorflow/`.
2. **Filtrado**: Excluye letras que requieren movimiento dinámico (`J` y `Z`).
3. **Extracción y normalización**:
   - Pasa cada muestra por el detector `HandLandmarker`. Si la detección directa falla, realiza un recorte con margen sobre el bounding box anotado.
   - Resta las coordenadas de la muñeca para centrar en el origen `(0, 0, 0)` y escala dividiendo por el valor absoluto máximo.
4. **Exportación tabular**: Guarda la matriz normalizada de 63 columnas (21 nodos × 3 ejes) en `data/lsch_coordenadas_estaticas.csv`.
5. **Clasificación**: Prepara los subconjuntos de entrenamiento y validación para entrenar un modelo clasificador (CNN 1D).

Para ejecutarlo:
```bash
jupyter notebook notebooks/Preprocesamiento_Entrenamiento.ipynb
```
