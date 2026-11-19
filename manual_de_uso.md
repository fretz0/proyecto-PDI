## Uso y Ejecución

### 1. Reconocimiento Continuo en Tiempo Real (`src/main.py`) — **Principal**

Ejecuta el sistema híbrido completo de 50 clases:

```bash
# Linux
.venv/bin/python src/main.py

# Windows
.venv\Scripts\python src\main.py
```

Opciones por línea de comandos:
```bash
# Linux / Windows (con el entorno activado)
python src/main.py --camera 0        # Seleccionar índice de cámara web (default: 0)
python src/main.py --threshold 0.65  # Ajustar umbral mínimo de confirmación (default: 0.65)
```

#### Controles interactivos en la ventana HUD:
- **`Q` o `Esc`**: Salir de la aplicación.
- **`C`**: Limpiar toda la oración acumulada.
- **`Retroceso (Backspace)`**: Borrar la última seña agregada a la oración.

#### Vocabulario reconocido (50 clases):
| Categoría | Clases Soportadas |
|---|---|
| **Alfabeto Estático (21)** | `A`, `B`, `C`, `D`, `E`, `F`, `H`, `I`, `K`, `L`, `M`, `N`, `O`, `P`, `R`, `T`, `U`, `V`, `W`, `Y`, `Q` |
| **Alfabeto Dinámico (6)** | **`G`**, **`J`**, **`Ñ`**, **`S`**, **`X`**, **`Z`** |
| **Palabras Unimanuales (9)** | **`Hola`**, **`Chao`**, **`Comer`**, **`Caminar`**, **`Flaco`**, **`Alto`**, **`Bajo`**, **`Bonito`**, **`Hablar`** |
| **Palabras Bimanuales (13)** | **`Ayudar`**, **`Abrir`**, **`Disculpar`**, **`Vender`**, **`Dibujar`**, **`Planchar`**, **`Amigo`**, **`Responsable`**, **`Arrendar`**, **`Chocar`**, **`Vestir`**, **`Estudiar`**, **`Jugar`** |
| **Estado Neutro (1)** | **`NONE`** (reposo / ausencia de gesto, con 0.0% de alucinación por movimiento de torso) |

---

### 2. Reconocimiento Estático Dactilológico (`src/main_estatico.py`)

Modo dedicado exclusivamente a las 24 letras estáticas mediante `HandLandmarker` y la red neuronal densa `models/mlp_lsch.keras`:

```bash
# Linux
.venv/bin/python src/main_estatico.py

# Windows
.venv\Scripts\python src\main_estatico.py
```

---

### 3. Notebooks de Análisis Estadístico y Entrenamiento (`notebooks/`)

1. **[`notebooks/Analisis_Estadistico_y_Tendencias_LSCh.ipynb`](notebooks/Analisis_Estadistico_y_Tendencias_LSCh.ipynb)** *(Nuevo — Auto-explicativo)*:
   - Diagnóstico matemático de la dominancia energética del torso ($96.8\%$ vs. $3.2\%$).
   - Estadísticas descriptivas de los datasets estático (`2,349 muestras`) y dinámico (`2,208 secuencias`, `50 clases`).
   - Mapas de separabilidad cinemática ($\Delta\text{Muñeca}$ vs. $\Delta\text{Dedos}$) y perfiles temporales del Diccionario LSCh.
   - Análisis de continuidad bimanual en contacto (`RobustHandTracker`), comparativa multiescenario, matriz de confusión y benchmark de latencia en GPU.

2. **[`notebooks/Preprocesamiento_Entrenamiento.ipynb`](notebooks/Preprocesamiento_Entrenamiento.ipynb)**:
   - Pipeline de extracción de coordenadas desde imágenes, normalización centrada en muñeca y entrenamiento del modelo `models/mlp_lsch.keras`.

Para abrir los notebooks interactivamente:
```bash
# Linux
.venv/bin/jupyter notebook notebooks/
