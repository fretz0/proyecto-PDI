# Reconocimiento de Lengua de Señas Chilena (LSCh) - PDI

Sistema de visión por computador para el reconocimiento en tiempo real de la Lengua de Señas Chilena (LSCh), con soporte tanto para el **alfabeto completo (incluyendo letras con movimiento)** como para **palabras y expresiones cotidianas**.

## Video de demostración

- [Demostración del sistema en Google Drive](https://drive.google.com/file/d/1eVWueIFO9i3UL1kuoAqHCHu4M0GUljVG/view?usp=sharing)

## Estructura del repositorio

```text
.
├── data/
│   ├── Senas de LSCh.v5i.tensorflow/     # Imágenes y anotaciones del dataset estático (train, valid, test)
│   └── lsch_coordenadas_estaticas.csv    # Coordenadas normalizadas estáticas (63 features + label + split)
├── models/
│   ├── gru_lsch_dinamico.h5              # Modelo GRU secuencial para reconocimiento dinámico (50 clases LSCh)
│   ├── holistic_landmarker.task          # Modelo oficial Holistic Landmarker de MediaPipe (cuerpo + 2 manos)
│   ├── hand_landmarker.task              # Archivo de modelo Hand Landmarker de MediaPipe
│   └── mlp_lsch.keras                    # Modelo MLP entrenado para clasificación de letras estáticas
├── notebooks/
│   └── Preprocesamiento_Entrenamiento.ipynb # Pipeline de extracción estática, entrenamiento MLP y evaluación
├── src/
│   ├── main.py                           # Reconocimiento dinámico en tiempo real (50 clases: letras + palabras)
│   └── main_estatico.py                  # Reconocimiento estático original (24 letras con MLP)
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

4. Modelos en la carpeta `models/`:
   - `models/gru_lsch_dinamico.h5`: Red neuronal recurrente (GRU) entrenada para 50 clases temporales (abecedario completo y expresiones de LSCh).
   - `models/holistic_landmarker.task`: Detector oficial de MediaPipe Tasks para postura y ambas manos simultáneas.
   - `models/mlp_lsch.keras`: Red neuronal densa (MLP) para las 24 letras estáticas.
   - `models/hand_landmarker.task`: Detector de landmarks de una sola mano para el modo estático.

---

## Uso y ejecución

### 1. Reconocimiento Dinámico en tiempo real (`src/main.py`) — **Recomendado**

Utiliza MediaPipe Holistic (postura y ambas manos) extrayendo un vector de 258 características por frame a través de una ventana deslizante de 30 frames ($\approx$ 1 segundo). Permite reconocer **tanto señas estáticas como señas con movimiento** y construir oraciones de señas en vivo:

```bash
python src/main.py
```

Opciones disponibles por línea de comandos:
```bash
python src/main.py --camera 0        # Forzar índice de cámara (ej. 0 para cámara integrada)
python src/main.py --threshold 0.80  # Ajustar umbral de confianza mínimo (por defecto 0.75)
```

#### Controles en vivo:
- **`q`**: Salir del programa.
- **`c`**: Limpiar el historial de señas reconocidas.

#### Clases reconocidas (50 clases):
| Categoría | Clases Soportadas |
|---|---|
| **Alfabeto Completo (27)** | `A`, `B`, `C`, `D`, `E`, `F`, `H`, `I`, `K`, `L`, `M`, `N`, `O`, `P`, `R`, `T`, `U`, `V`, `W`, `Y`, `Q` y las dinámicas con movimiento: **`G`**, **`J`**, **`Ñ`**, **`S`**, **`X`**, **`Z`**. |
| **Palabras y Expresiones (22)** | **`Hola`**, **`Chao`**, **`Ayudar`**, **`Disculpar`** (disculpa / por favor), **`Comer`**, **`Caminar`**, **`Flaco`**, **`Alto`**, **`Bajo`**, **`Bonito`**, **`Hablar`**, **`Abrir`**, **`Vender`**, **`Dibujar`**, **`Planchar`**, **`Amigo`**, **`Responsable`**, **`Arrendar`**, **`Chocar`**, **`Vestir`**, **`Estudiar`**, **`Jugar`**. |
| **Estado Neutro (1)** | **`NONE`** (reposo / ausencia de gesto, evitando falsos positivos continuos). |

---

### 2. Reconocimiento Estático Tradicional (`src/main_estatico.py`)

Inicia el detector de una sola mano con `HandLandmarker` y clasifica postura manual cuadro a cuadro con el modelo MLP (`models/mlp_lsch.keras`), desplegando el grafo esquelético y el panel de coordenadas numéricas normalizadas (24 clases estáticas, excluyendo señas dinámicas).

```bash
python src/main_estatico.py
```

---

### 3. Pipeline de Entrenamiento Estático (`notebooks/Preprocesamiento_Entrenamiento.ipynb`)

Contiene el flujo de extracción y entrenamiento del clasificador MLP sobre el dataset de Roboflow (`Senas de LSCh.v5i.tensorflow`):
1. **Lectura y recorte**: Carga imágenes y bounding boxes con margen de 20 píxeles.
2. **Normalización**: Centrado en muñeca $(0, 0, 0)$ y escalado por valor absoluto máximo.
3. **Exportación**: Almacena `data/lsch_coordenadas_estaticas.csv` (63 coordenadas normalizadas).
4. **Entrenamiento MLP**: Capas densas (128 y 64 con Dropout) entrenadas por 80 épocas con ponderación de clases (`compute_class_weight`).
5. **Evaluación**: Matriz de confusión y guardado en `models/mlp_lsch.keras`.

Para abrir el notebook:
```bash
jupyter notebook notebooks/Preprocesamiento_Entrenamiento.ipynb
```
