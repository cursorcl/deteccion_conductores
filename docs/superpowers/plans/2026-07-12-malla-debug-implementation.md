# Visualización de Malla Facial de Debug — Plan de Implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Agregar un flag `--malla` a `dsd/main.py` que dibuja la malla facial completa de Mediapipe (478 puntos + líneas de teselación) y resalta los puntos de control que la app ya usa para detección (ojos, iris, boca).

**Architecture:** `dsd/face_mesh.py` expone los 478 landmarks crudos además del subconjunto curado ya existente; un nuevo módulo `dsd/debug_draw.py` dibuja esa malla usando la tabla de conexiones de teselación que ya trae la librería Mediapipe instalada (`FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION`); `dsd/main.py` gana un flag `argparse` que activa esa función de dibujo en vez del overlay actual (solo puntos de ojo).

**Tech Stack:** Python 3.11, pytest, OpenCV, Mediapipe Face Landmarker (Tasks API).

## Global Constraints

- No se hardcodea ninguna tabla de conexiones propia — se usa `FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION`, ya disponible en `mediapipe.tasks.python.vision.face_landmarker` en la versión instalada (`mediapipe==0.10.35`).
- La detección de landmarks (con o sin `--malla`) solo corre dentro del bloque `Estado.ACTIVA` ya existente — no se activa detección de rostro fuera de ese caso.
- Sin el flag `--malla`, el comportamiento de `dsd/main.py` debe ser idéntico al actual (solo puntos de ojo en amarillo).
- Colores: `COLOR_MALLA = (120, 120, 120)` (gris, líneas y puntos de fondo), `COLOR_CONTROL = (0, 255, 255)` (amarillo, puntos de control — mismo color que ya usan los ojos hoy).
- `dsd/db.py` no se modifica — esto es puro dibujo, sin persistencia.

---

## Task 1: `dsd/face_mesh.py` — exponer los 478 landmarks crudos

**Files:**
- Modify: `dsd/face_mesh.py`

**Interfaces:**
- Produces: `ResultadoLandmarks.puntos_todos: List[Tuple[float, float]]` — los 478 landmarks crudos de Mediapipe, en orden (índice 0 a 477), en las mismas coordenadas de píxel que el resto de los campos (`puntos_ojo_derecho`, etc.).

No hay test automatizado para este archivo (depende de cámara/Mediapipe real — mismo criterio ya aplicado a los demás campos de `ResultadoLandmarks`, sin `tests/test_face_mesh.py` en el proyecto). Verificación manual en Step 3.

- [ ] **Step 1: Reemplazar el contenido completo de `dsd/face_mesh.py`**

```python
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# Indices fijos de la topologia de landmarks de Mediapipe Face Landmarker
# correspondientes al contorno de cada ojo, en el orden estandar de 6 puntos
# para el calculo de EAR (Eye Aspect Ratio) segun Soukupova & Cech, 2016:
# [esquina_externa, parpado_superior_1, parpado_superior_2, esquina_interna,
#  parpado_inferior_2, parpado_inferior_1]
INDICES_OJO_DERECHO = [33, 160, 158, 133, 153, 144]
INDICES_OJO_IZQUIERDO = [362, 385, 387, 263, 373, 380]

# Indices del centro del iris (topologia de 478 puntos: 468 base + 10 de
# iris, 5 por ojo -- 1 centro + 4 puntos de contorno). Estos indices se
# verifican manualmente en este mismo task (Step 2) contra fotos reales
# antes de darlos por buenos.
INDICE_IRIS_DERECHO = 468
INDICE_IRIS_IZQUIERDO = 473

# Indices verificados manualmente contra camara real (mismo proceso que
# INDICE_IRIS_DERECHO/IZQUIERDO) para el calculo de MAR (Mouth Aspect
# Ratio), en el mismo orden de 6 puntos que EAR: [comisura_izquierda,
# labio_superior_izquierdo, labio_superior_derecho, comisura_derecha,
# labio_inferior_derecho, labio_inferior_izquierdo].
INDICES_BOCA = [61, 40, 270, 291, 314, 84]

RUTA_MODELO = "models/face_landmarker.task"

# El detector se crea una sola vez al importar el modulo (no en cada
# llamada) porque instanciarlo (cargar el modelo) es costoso; la inferencia
# por-frame en modo IMAGE es liviana en comparacion.
_base_options = BaseOptions(model_asset_path=RUTA_MODELO)
_options = vision.FaceLandmarkerOptions(
    base_options=_base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=True,
)
_detector = vision.FaceLandmarker.create_from_options(_options)


@dataclass
class ResultadoLandmarks:
    puntos_ojo_derecho: List[Tuple[float, float]]
    puntos_ojo_izquierdo: List[Tuple[float, float]]
    iris_derecho: Tuple[float, float]
    iris_izquierdo: Tuple[float, float]
    matriz_rotacion: List[List[float]]
    puntos_boca: List[Tuple[float, float]]
    puntos_todos: List[Tuple[float, float]]


def detectar_landmarks(frame) -> Optional[ResultadoLandmarks]:
    try:
        alto, ancho = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        resultado = _detector.detect(imagen_mp)
    except Exception:
        return None

    if not resultado.face_landmarks or not resultado.facial_transformation_matrixes:
        return None

    landmarks = resultado.face_landmarks[0]

    def punto(indice: int) -> Tuple[float, float]:
        lm = landmarks[indice]
        return (lm.x * ancho, lm.y * alto)

    puntos_ojo_derecho = [punto(i) for i in INDICES_OJO_DERECHO]
    puntos_ojo_izquierdo = [punto(i) for i in INDICES_OJO_IZQUIERDO]
    iris_derecho = punto(INDICE_IRIS_DERECHO)
    iris_izquierdo = punto(INDICE_IRIS_IZQUIERDO)
    puntos_boca = [punto(i) for i in INDICES_BOCA]
    puntos_todos = [punto(i) for i in range(len(landmarks))]

    matriz_4x4 = resultado.facial_transformation_matrixes[0]
    matriz_rotacion = [[float(matriz_4x4[i][j]) for j in range(3)] for i in range(3)]

    return ResultadoLandmarks(
        puntos_ojo_derecho=puntos_ojo_derecho,
        puntos_ojo_izquierdo=puntos_ojo_izquierdo,
        iris_derecho=iris_derecho,
        iris_izquierdo=iris_izquierdo,
        matriz_rotacion=matriz_rotacion,
        puntos_boca=puntos_boca,
        puntos_todos=puntos_todos,
    )
```

- [ ] **Step 2: Confirmar que el resto de la suite sigue pasando**

Run: `pytest -v`
Expected: todos los tests existentes PASS (nada más en el repo construye `ResultadoLandmarks` con argumentos posicionales, así que agregar un campo al final no rompe nada).

- [ ] **Step 3: Verificación manual (cámara real)**

```bash
source .venv/bin/activate
python -c "
import cv2
from dsd.face_mesh import detectar_landmarks

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
resultado = detectar_landmarks(frame)
if resultado is None:
    print('sin rostro detectado')
else:
    print('cantidad de puntos_todos:', len(resultado.puntos_todos))
    print('primer punto:', resultado.puntos_todos[0])
cap.release()
"
```
Expected: `cantidad de puntos_todos: 478` (con la cara encuadrada frente a la cámara).

- [ ] **Step 4: Commit**

```bash
git add dsd/face_mesh.py
git commit -m "feat: exponer los 478 landmarks crudos en ResultadoLandmarks"
```

---

## Task 2: `dsd/debug_draw.py` — dibujo de la malla completa

**Files:**
- Create: `dsd/debug_draw.py`
- Test: `tests/test_debug_draw.py`

**Interfaces:**
- Consumes: `ResultadoLandmarks` de `dsd/face_mesh.py` (Task 1) — usa `puntos_todos`, `puntos_ojo_derecho`, `puntos_ojo_izquierdo`, `puntos_boca`, `iris_derecho`, `iris_izquierdo`.
- Produces: `dibujar_malla_debug(frame, landmarks: ResultadoLandmarks) -> None` — dibuja in-place sobre `frame` (mismo patrón que las llamadas a `cv2.circle`/`cv2.putText` ya existentes en `dsd/main.py`), sin valor de retorno.

- [ ] **Step 1: Escribir el test que falla**

Crear `tests/test_debug_draw.py`:

```python
from dsd.debug_draw import CONEXIONES_TESELACION


def test_conexiones_teselacion_no_esta_vacia():
    assert len(CONEXIONES_TESELACION) > 0


def test_conexiones_teselacion_indices_en_rango_valido():
    # Los 478 landmarks de Mediapipe van de indice 0 a 477; si una version
    # futura de Mediapipe cambia esta tabla, este test debe fallar en vez
    # de dejar pasar un indice fuera de rango silenciosamente (que causaria
    # un IndexError recien al dibujar con una camara real).
    for conexion in CONEXIONES_TESELACION:
        assert 0 <= conexion.start <= 477
        assert 0 <= conexion.end <= 477
```

- [ ] **Step 2: Confirmar que falla**

Run: `pytest tests/test_debug_draw.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dsd.debug_draw'`

- [ ] **Step 3: Implementar `dsd/debug_draw.py`**

```python
from typing import TYPE_CHECKING

import cv2
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

if TYPE_CHECKING:
    from dsd.face_mesh import ResultadoLandmarks

# Tabla de conexiones de la triangulacion real de la malla de Mediapipe
# (478 puntos), ya incluida en la libreria instalada -- no se hardcodea
# ninguna tabla propia.
CONEXIONES_TESELACION = FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION

COLOR_MALLA = (120, 120, 120)  # gris: lineas y puntos de fondo de la malla
COLOR_CONTROL = (0, 255, 255)  # amarillo: puntos de control que usa la deteccion


def dibujar_malla_debug(frame, landmarks: "ResultadoLandmarks") -> None:
    """Dibuja sobre `frame` (in-place): la malla completa de 478 puntos +
    lineas de teselacion en gris, y los puntos de control que la app usa
    para deteccion (ojos, iris, boca) resaltados en amarillo."""
    for conexion in CONEXIONES_TESELACION:
        punto_inicio = landmarks.puntos_todos[conexion.start]
        punto_fin = landmarks.puntos_todos[conexion.end]
        cv2.line(
            frame,
            (int(punto_inicio[0]), int(punto_inicio[1])),
            (int(punto_fin[0]), int(punto_fin[1])),
            COLOR_MALLA,
            1,
        )

    for x, y in landmarks.puntos_todos:
        cv2.circle(frame, (int(x), int(y)), 1, COLOR_MALLA, -1)

    puntos_control = (
        landmarks.puntos_ojo_derecho
        + landmarks.puntos_ojo_izquierdo
        + landmarks.puntos_boca
        + [landmarks.iris_derecho, landmarks.iris_izquierdo]
    )
    for x, y in puntos_control:
        cv2.circle(frame, (int(x), int(y)), 2, COLOR_CONTROL, -1)
```

- [ ] **Step 4: Confirmar que pasa**

Run: `pytest tests/test_debug_draw.py -v`
Expected: 2 tests PASS

- [ ] **Step 5: Confirmar que el resto de la suite sigue pasando**

Run: `pytest -v`
Expected: todos PASS

- [ ] **Step 6: Commit**

```bash
git add dsd/debug_draw.py tests/test_debug_draw.py
git commit -m "feat: agregar dibujo de malla facial de debug"
```

---

## Task 3: `dsd/main.py` — flag `--malla` e integración

**Files:**
- Modify: `dsd/main.py`

**Interfaces:**
- Consumes: `dibujar_malla_debug` (Task 2), `ResultadoLandmarks.puntos_todos` (Task 1).
- Produces: `main(mostrar_malla: bool = False) -> None` (antes `main() -> None`, sin parámetros).

No hay test automatizado para `dsd/main.py` (mismo criterio que el resto del archivo: requiere cámara real). Verificación manual en Step 4.

- [ ] **Step 1: Agregar el import**

En `dsd/main.py`, agregar junto a los demás imports de `dsd.*` (después de `from dsd.db import (...)`, junto a los otros imports de un solo nombre — no importa el orden exacto respecto a los demás, ya que el archivo no tiene una herramienta de lint que lo exija):

```python
from dsd.debug_draw import dibujar_malla_debug
```

Y al principio del archivo, junto a `import threading` / `import time`, agregar:

```python
import argparse
```

- [ ] **Step 2: Cambiar la firma de `main` y el bloque de dibujo**

Reemplazar:

```python
def main() -> None:
    global frame_actual
```

por:

```python
def main(mostrar_malla: bool = False) -> None:
    global frame_actual
```

Reemplazar:

```python
                if landmarks is not None:
                    for x, y in landmarks.puntos_ojo_derecho + landmarks.puntos_ojo_izquierdo:
                        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)
```

por:

```python
                if landmarks is not None:
                    if mostrar_malla:
                        dibujar_malla_debug(frame, landmarks)
                    else:
                        for x, y in landmarks.puntos_ojo_derecho + landmarks.puntos_ojo_izquierdo:
                            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)
```

- [ ] **Step 3: Agregar el parseo de argumentos**

Reemplazar:

```python
if __name__ == "__main__":
    main()
```

por:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deteccion de somnolencia y distraccion del conductor."
    )
    parser.add_argument(
        "--malla",
        action="store_true",
        help=(
            "Dibuja la malla facial completa (478 puntos + lineas de "
            "teselacion) y resalta los puntos de control de deteccion "
            "(ojos, iris, boca), en vez de la superposicion normal."
        ),
    )
    args = parser.parse_args()
    main(mostrar_malla=args.malla)
```

- [ ] **Step 4: Confirmar que la suite completa sigue pasando**

Run: `pytest -v`
Expected: todos los tests PASS

- [ ] **Step 5: Verificación manual (cámara real)**

```bash
source .venv/bin/activate
python -m dsd.main --malla
```

Con una sesión activa (conductor ya enrolado y reconocido):
1. Debe verse la malla completa (puntos y líneas grises) siguiendo la cara al moverse, más los puntos de control (ojos, iris, boca) resaltados en amarillo sobre esa malla.
2. Salir con `q`.
3. Correr sin el flag: `python -m dsd.main` — debe verse exactamente igual que antes de este cambio (solo puntos de ojo en amarillo, sin malla ni líneas).

- [ ] **Step 6: Commit**

```bash
git add dsd/main.py
git commit -m "feat: agregar flag --malla para visualizar la malla facial completa"
```

---

## Self-Review

**Cobertura de la spec:**
- `puntos_todos` (478 landmarks crudos) en `ResultadoLandmarks` → Task 1. ✓
- Malla completa (puntos grises + líneas de teselación real vía `FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION`, sin hardcodear tabla propia) → Task 2. ✓
- Puntos de control (ojos, iris, boca) resaltados en amarillo → Task 2 (`dibujar_malla_debug`). ✓
- Flag `--malla` vía `argparse`, mismo patrón que `dsd/enroll.py` → Task 3. ✓
- Sin el flag, comportamiento idéntico al actual → Task 3, Step 2 (rama `else` conserva el código original) y Step 5 (verificación manual explícita de ambos casos). ✓
- Detección solo dentro de `Estado.ACTIVA` → sin cambios a esa condición en ningún task. ✓
- Smoke test de la tabla de conexiones externa (protección ante cambio de versión de Mediapipe) → Task 2. ✓
- Fuera de alcance (detección fuera de `Estado.ACTIVA`, indicador en pantalla, cambios a somnolencia/distracción/reconocimiento, persistencia de capturas) → ningún task los incluye. ✓

Sin gaps encontrados.
