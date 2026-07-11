# Detección de Bostezo — Plan de Implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Agregar detección de bostezo (evento individual `bostezo` + evento agregado de frecuencia `fatiga_bostezos`) al detector de somnolencia existente.

**Architecture:** Nuevo `dsd/mouth_metrics.py` (MAR, geométricamente idéntico al EAR) alimentado por nuevos puntos de boca en `dsd/face_mesh.py`; `dsd/drowsiness_state.py` gana un tercer temporizador de sostenimiento (reutilizando `dsd/sustained_timer.py`, igual que microsueño) más una ventana deslizante de conteo de ocurrencias (mismo principio que PERCLOS, pero contando eventos discretos). Nuevos umbrales en `config/somnolencia.yaml`. `dsd/db.py` no cambia (tabla `events` ya genérica).

**Tech Stack:** Python 3.11, pytest, PyYAML, Mediapipe Face Landmarker, OpenCV.

## Global Constraints

- Índices de landmarks de boca (`INDICES_BOCA = [61, 40, 270, 291, 314, 84]`) ya verificados manualmente contra cámara real — no se re-verifican en este plan.
- Reutilizar `dsd/sustained_timer.py` (`EstadoTemporizadorSostenido`, `procesar_temporizador_sostenido`) para el temporizador de bostezo individual — no crear un temporizador nuevo.
- Reutilizar los campos `cooldown_segundos` y `gap_maximo_segundos` ya existentes en `ConfigSomnolencia` — no crear campos de cooldown/gap separados para bostezo.
- `dsd/db.py` y el esquema de la tabla `events` no se modifican.
- Sin alertas audio/visuales — solo detección + persistencia (mismo alcance que microsueño/PERCLOS/distracción).

---

## Task 1: `dsd/mouth_metrics.py` — cálculo de MAR (puro)

**Files:**
- Create: `dsd/mouth_metrics.py`
- Test: `tests/test_mouth_metrics.py`

**Interfaces:**
- Produces: `calcular_mar(puntos_boca: Sequence[Tuple[float, float]]) -> float`, misma firma/contrato que `calcular_ear` en `dsd/eye_metrics.py` (6 puntos, `ValueError` si no son exactamente 6).

- [ ] **Step 1: Escribir el test que falla**

Crear `tests/test_mouth_metrics.py`:

```python
import pytest

from dsd.mouth_metrics import calcular_mar

# Puntos sintéticos en el orden estándar de 6 puntos MAR:
# (comisura_izquierda, labio_superior_izquierdo, labio_superior_derecho,
#  comisura_derecha, labio_inferior_derecho, labio_inferior_izquierdo)
BOCA_CERRADA = [
    (0.0, 0.0), (0.3, -0.05), (0.7, -0.05), (1.0, 0.0), (0.7, 0.05), (0.3, 0.05)
]
BOCA_ABIERTA = [
    (0.0, 0.0), (0.3, -0.4), (0.7, -0.4), (1.0, 0.0), (0.7, 0.4), (0.3, 0.4)
]


def test_boca_cerrada_da_mar_bajo():
    assert calcular_mar(BOCA_CERRADA) == pytest.approx(0.1)


def test_boca_abierta_da_mar_alto():
    assert calcular_mar(BOCA_ABIERTA) == pytest.approx(0.8)


def test_mar_es_invariante_a_la_escala():
    boca_escalada = [(x * 100.0, y * 100.0) for x, y in BOCA_ABIERTA]
    assert calcular_mar(boca_escalada) == pytest.approx(calcular_mar(BOCA_ABIERTA))


def test_boca_abierta_supera_umbral_tipico():
    assert calcular_mar(BOCA_ABIERTA) > 0.6


def test_boca_cerrada_no_supera_umbral_tipico():
    assert calcular_mar(BOCA_CERRADA) < 0.6


def test_calcular_mar_lanza_error_si_no_son_6_puntos():
    with pytest.raises(ValueError):
        calcular_mar([(0.0, 0.0), (1.0, 1.0)])
```

- [ ] **Step 2: Confirmar que falla**

Run: `pytest tests/test_mouth_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dsd.mouth_metrics'`

- [ ] **Step 3: Implementar `dsd/mouth_metrics.py`**

```python
import math
from typing import Sequence, Tuple


def calcular_mar(puntos_boca: Sequence[Tuple[float, float]]) -> float:
    """Calcula el Mouth Aspect Ratio (MAR), geometricamente identico al
    EAR (ver dsd/eye_metrics.py) pero aplicado a los 6 puntos de boca en
    el orden estandar: (comisura_izquierda, labio_superior_izquierdo,
    labio_superior_derecho, comisura_derecha, labio_inferior_derecho,
    labio_inferior_izquierdo).

    MAR = (dist(p2, p6) + dist(p3, p5)) / (2 * dist(p1, p4))

    A diferencia del EAR (donde el ojo cerrado da el valor bajo), aqui la
    boca cerrada da MAR bajo y la boca abierta (bostezo) da MAR alto.
    """
    if len(puntos_boca) != 6:
        raise ValueError(
            "calcular_mar requiere exactamente 6 puntos (comisura_izquierda, "
            "labio_superior_izquierdo, labio_superior_derecho, comisura_derecha, "
            f"labio_inferior_derecho, labio_inferior_izquierdo); se recibieron {len(puntos_boca)}."
        )

    p1, p2, p3, p4, p5, p6 = puntos_boca

    def distancia(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    vertical = distancia(p2, p6) + distancia(p3, p5)
    horizontal = distancia(p1, p4)

    if horizontal == 0.0:
        return 0.0

    return vertical / (2 * horizontal)
```

- [ ] **Step 4: Confirmar que pasa**

Run: `pytest tests/test_mouth_metrics.py -v`
Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add dsd/mouth_metrics.py tests/test_mouth_metrics.py
git commit -m "feat: agregar calculo de MAR (Mouth Aspect Ratio)"
```

---

## Task 2: `dsd/face_mesh.py` — puntos de boca

**Files:**
- Modify: `dsd/face_mesh.py`

**Interfaces:**
- Consumes: nada nuevo (usa el mismo `_detector` ya existente).
- Produces: `ResultadoLandmarks.puntos_boca: List[Tuple[float, float]]` (6 puntos, mismo orden que `calcular_mar` espera).

No hay test automatizado para este archivo (depende de cámara/Mediapipe real — mismo criterio ya aplicado a `puntos_ojo_*`/`iris_*`, sin `tests/test_face_mesh.py` en el proyecto). La verificación es manual (Step 3).

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

    matriz_4x4 = resultado.facial_transformation_matrixes[0]
    matriz_rotacion = [[float(matriz_4x4[i][j]) for j in range(3)] for i in range(3)]

    return ResultadoLandmarks(
        puntos_ojo_derecho=puntos_ojo_derecho,
        puntos_ojo_izquierdo=puntos_ojo_izquierdo,
        iris_derecho=iris_derecho,
        iris_izquierdo=iris_izquierdo,
        matriz_rotacion=matriz_rotacion,
        puntos_boca=puntos_boca,
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
print('puntos_boca:', resultado.puntos_boca if resultado else 'sin rostro detectado')
cap.release()
"
```
Expected: imprime una lista de 6 tuplas `(x, y)` con valores dentro del tamaño del frame (no `sin rostro detectado`, con la cara encuadrada).

- [ ] **Step 4: Commit**

```bash
git add dsd/face_mesh.py
git commit -m "feat: agregar puntos de boca a ResultadoLandmarks"
```

---

## Task 3: `dsd/config.py` + `config/somnolencia.yaml` — umbrales de bostezo

**Files:**
- Modify: `dsd/config.py`
- Modify: `config/somnolencia.yaml`
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces: `ConfigSomnolencia` gana los campos `mar_umbral`, `bostezo_min_segundos`, `bostezo_ventana_segundos`, `bostezo_umbral_cantidad` (todos `float`), poblados por `cargar_config`.

- [ ] **Step 1: Escribir el test que falla**

Reemplazar el contenido completo de `tests/test_config.py`:

```python
import pytest

from dsd.config import ConfigSomnolencia, cargar_config
from dsd.config import ConfigDistraccion, cargar_config_distraccion

YAML_VALIDO = """
ear_umbral: 0.21
microsueno_segundos: 1.5
perclos_ventana_segundos: 60
perclos_umbral: 0.15
cooldown_segundos: 30
perclos_cobertura_minima: 0.5
gap_maximo_segundos: 1.0
mar_umbral: 0.6
bostezo_min_segundos: 1.5
bostezo_ventana_segundos: 300
bostezo_umbral_cantidad: 3
"""


def test_cargar_config_retorna_los_valores_correctos(tmp_path):
    ruta = tmp_path / "somnolencia.yaml"
    ruta.write_text(YAML_VALIDO)

    config = cargar_config(str(ruta))

    assert config == ConfigSomnolencia(
        ear_umbral=0.21,
        microsueno_segundos=1.5,
        perclos_ventana_segundos=60.0,
        perclos_umbral=0.15,
        cooldown_segundos=30.0,
        perclos_cobertura_minima=0.5,
        gap_maximo_segundos=1.0,
        mar_umbral=0.6,
        bostezo_min_segundos=1.5,
        bostezo_ventana_segundos=300.0,
        bostezo_umbral_cantidad=3.0,
    )


def test_cargar_config_convierte_valores_a_float(tmp_path):
    ruta = tmp_path / "somnolencia.yaml"
    ruta.write_text(YAML_VALIDO)

    config = cargar_config(str(ruta))

    assert isinstance(config.perclos_ventana_segundos, float)
    assert isinstance(config.cooldown_segundos, float)
    assert isinstance(config.bostezo_umbral_cantidad, float)


def test_cargar_config_clave_faltante_lanza_keyerror(tmp_path):
    ruta = tmp_path / "somnolencia.yaml"
    ruta.write_text("ear_umbral: 0.21\n")

    with pytest.raises(KeyError):
        cargar_config(str(ruta))


def test_cargar_config_archivo_real_del_proyecto():
    config = cargar_config("config/somnolencia.yaml")
    assert config.ear_umbral == 0.21
    assert config.microsueno_segundos == 1.5
    assert config.perclos_ventana_segundos == 60.0
    assert config.perclos_umbral == 0.15
    assert config.cooldown_segundos == 30.0
    assert config.perclos_cobertura_minima == 0.5
    assert config.gap_maximo_segundos == 1.0
    assert config.mar_umbral == 0.6
    assert config.bostezo_min_segundos == 1.5
    assert config.bostezo_ventana_segundos == 300.0
    assert config.bostezo_umbral_cantidad == 3.0


YAML_VALIDO_DISTRACCION = """
distraccion_segundos: 2.0
yaw_umbral_grados: 20
pitch_umbral_grados: 15
gaze_ratio_umbral: 0.20
cooldown_segundos: 30
gap_maximo_segundos: 1.0
"""


def test_cargar_config_distraccion_retorna_los_valores_correctos(tmp_path):
    ruta = tmp_path / "distraccion.yaml"
    ruta.write_text(YAML_VALIDO_DISTRACCION)

    config = cargar_config_distraccion(str(ruta))

    assert config == ConfigDistraccion(
        distraccion_segundos=2.0,
        yaw_umbral_grados=20.0,
        pitch_umbral_grados=15.0,
        gaze_ratio_umbral=0.20,
        cooldown_segundos=30.0,
        gap_maximo_segundos=1.0,
    )


def test_cargar_config_distraccion_convierte_valores_a_float(tmp_path):
    ruta = tmp_path / "distraccion.yaml"
    ruta.write_text(YAML_VALIDO_DISTRACCION)

    config = cargar_config_distraccion(str(ruta))

    assert isinstance(config.yaw_umbral_grados, float)
    assert isinstance(config.cooldown_segundos, float)


def test_cargar_config_distraccion_clave_faltante_lanza_keyerror(tmp_path):
    ruta = tmp_path / "distraccion.yaml"
    ruta.write_text("distraccion_segundos: 2.0\n")

    with pytest.raises(KeyError):
        cargar_config_distraccion(str(ruta))


def test_cargar_config_distraccion_archivo_real_del_proyecto():
    config = cargar_config_distraccion("config/distraccion.yaml")
    assert config.distraccion_segundos == 2.0
    assert config.yaw_umbral_grados == 20.0
    assert config.pitch_umbral_grados == 15.0
    assert config.gaze_ratio_umbral == 0.20
    assert config.cooldown_segundos == 30.0
    assert config.gap_maximo_segundos == 1.0
```

- [ ] **Step 2: Confirmar que falla**

Run: `pytest tests/test_config.py -v`
Expected: FAIL en `test_cargar_config_retorna_los_valores_correctos` y otros — `TypeError: cargar_config() got an unexpected keyword argument 'mar_umbral'` (o `KeyError` en `test_cargar_config_archivo_real_del_proyecto`, porque el YAML real todavía no tiene las claves nuevas).

- [ ] **Step 3: Actualizar `dsd/config.py`**

Reemplazar el contenido completo del archivo:

```python
from dataclasses import dataclass

import yaml

CAMPOS_REQUERIDOS = (
    "ear_umbral",
    "microsueno_segundos",
    "perclos_ventana_segundos",
    "perclos_umbral",
    "cooldown_segundos",
    "perclos_cobertura_minima",
    "gap_maximo_segundos",
    "mar_umbral",
    "bostezo_min_segundos",
    "bostezo_ventana_segundos",
    "bostezo_umbral_cantidad",
)


@dataclass
class ConfigSomnolencia:
    ear_umbral: float
    microsueno_segundos: float
    perclos_ventana_segundos: float
    perclos_umbral: float
    cooldown_segundos: float
    perclos_cobertura_minima: float
    gap_maximo_segundos: float
    mar_umbral: float
    bostezo_min_segundos: float
    bostezo_ventana_segundos: float
    bostezo_umbral_cantidad: float


def cargar_config(path: str) -> ConfigSomnolencia:
    with open(path, "r", encoding="utf-8") as archivo:
        datos = yaml.safe_load(archivo)

    faltantes = [campo for campo in CAMPOS_REQUERIDOS if campo not in datos]
    if faltantes:
        raise KeyError(
            f"Faltan claves requeridas en el archivo de configuracion '{path}': {faltantes}"
        )

    return ConfigSomnolencia(
        ear_umbral=float(datos["ear_umbral"]),
        microsueno_segundos=float(datos["microsueno_segundos"]),
        perclos_ventana_segundos=float(datos["perclos_ventana_segundos"]),
        perclos_umbral=float(datos["perclos_umbral"]),
        cooldown_segundos=float(datos["cooldown_segundos"]),
        perclos_cobertura_minima=float(datos["perclos_cobertura_minima"]),
        gap_maximo_segundos=float(datos["gap_maximo_segundos"]),
        mar_umbral=float(datos["mar_umbral"]),
        bostezo_min_segundos=float(datos["bostezo_min_segundos"]),
        bostezo_ventana_segundos=float(datos["bostezo_ventana_segundos"]),
        bostezo_umbral_cantidad=float(datos["bostezo_umbral_cantidad"]),
    )


CAMPOS_REQUERIDOS_DISTRACCION = (
    "distraccion_segundos",
    "yaw_umbral_grados",
    "pitch_umbral_grados",
    "gaze_ratio_umbral",
    "cooldown_segundos",
    "gap_maximo_segundos",
)


@dataclass
class ConfigDistraccion:
    distraccion_segundos: float
    yaw_umbral_grados: float
    pitch_umbral_grados: float
    gaze_ratio_umbral: float
    cooldown_segundos: float
    gap_maximo_segundos: float


def cargar_config_distraccion(path: str) -> ConfigDistraccion:
    with open(path, "r", encoding="utf-8") as archivo:
        datos = yaml.safe_load(archivo)

    faltantes = [campo for campo in CAMPOS_REQUERIDOS_DISTRACCION if campo not in datos]
    if faltantes:
        raise KeyError(
            f"Faltan claves requeridas en el archivo de configuracion '{path}': {faltantes}"
        )

    return ConfigDistraccion(
        distraccion_segundos=float(datos["distraccion_segundos"]),
        yaw_umbral_grados=float(datos["yaw_umbral_grados"]),
        pitch_umbral_grados=float(datos["pitch_umbral_grados"]),
        gaze_ratio_umbral=float(datos["gaze_ratio_umbral"]),
        cooldown_segundos=float(datos["cooldown_segundos"]),
        gap_maximo_segundos=float(datos["gap_maximo_segundos"]),
    )
```

- [ ] **Step 4: Agregar los umbrales nuevos a `config/somnolencia.yaml`**

Agregar al final del archivo (después del bloque de `gap_maximo_segundos` ya existente):

```yaml

# Mouth Aspect Ratio (MAR): razon geometrica identica al EAR (ver
# dsd/mouth_metrics.py) pero aplicada a los 6 puntos de boca. Boca cerrada
# -> MAR bajo (~0.3-0.5 tipico); boca abierta en bostezo -> MAR alto
# (>1.0 tipico). Decision de ingenieria -- se calibrara con verificacion
# manual real, igual que gaze_ratio_umbral en el detector de distraccion.
mar_umbral: 0.6

# Duracion minima continua (segundos) con la boca abierta para contar como
# un bostezo (y no, p.ej., hablar o gesticular). Decision de ingenieria,
# sujeta a calibracion con pruebas reales.
bostezo_min_segundos: 1.5

# Tamano (segundos) de la ventana deslizante para contar frecuencia de
# bostezos. Decision de ingenieria, sujeta a calibracion con pruebas
# reales -- analoga a perclos_ventana_segundos pero para conteo de eventos
# discretos en vez de tiempo ponderado.
bostezo_ventana_segundos: 300

# Cantidad minima de bostezos dentro de la ventana anterior para
# considerar "fatiga por bostezos". Decision de ingenieria, sujeta a
# calibracion con pruebas reales.
bostezo_umbral_cantidad: 3
```

- [ ] **Step 5: Confirmar que pasa**

Run: `pytest tests/test_config.py -v`
Expected: todos los tests PASS

- [ ] **Step 6: Commit**

```bash
git add dsd/config.py config/somnolencia.yaml tests/test_config.py
git commit -m "feat: agregar umbrales de bostezo a ConfigSomnolencia"
```

---

## Task 4: `dsd/drowsiness_state.py` — renombrar a `procesar_somnolencia` + bostezo individual

**Files:**
- Modify: `dsd/drowsiness_state.py`
- Modify: `tests/test_drowsiness_state.py`

**Interfaces:**
- Consumes: `calcular_mar` no se usa directamente aquí (lo llama `dsd/main.py` en Task 6) — este módulo solo recibe `mar: float` ya calculado, igual que ya recibe `ear: float`.
- Produces: `procesar_somnolencia(estado, ear, mar, timestamp, config) -> (EstadoSomnolencia, list[EventoSomnolencia])` — reemplaza a `procesar_ear` (mismo nombre de función ya no existe). `EstadoSomnolencia` gana `boca_abierta_inicio`, `ultimo_disparo_bostezo`, `bostezos: List[float]`, `ultimo_disparo_fatiga_bostezos` (este último no se usa todavía en este task, se deja en `None` sin lógica — se activa en Task 5).

- [ ] **Step 1: Escribir el test que falla**

Reemplazar el contenido completo de `tests/test_drowsiness_state.py`:

```python
from dsd.config import ConfigSomnolencia
from dsd.drowsiness_state import (
    EstadoSomnolencia,
    EventoSomnolencia,
    Muestra,
    estado_inicial_somnolencia,
    procesar_somnolencia,
)

CONFIG = ConfigSomnolencia(
    ear_umbral=0.21,
    microsueno_segundos=1.5,
    perclos_ventana_segundos=60.0,
    perclos_umbral=0.15,
    cooldown_segundos=30.0,
    perclos_cobertura_minima=0.5,
    gap_maximo_segundos=1.0,
    mar_umbral=0.6,
    bostezo_min_segundos=1.5,
    bostezo_ventana_segundos=300.0,
    bostezo_umbral_cantidad=3.0,
)

EAR_CERRADO = 0.10
EAR_ABIERTO = 0.30
MAR_CERRADO = 0.10
MAR_ABIERTO = 0.80


def test_ojo_abierto_no_acumula_cierre():
    estado = estado_inicial_somnolencia()
    nuevo_estado, eventos = procesar_somnolencia(
        estado, EAR_ABIERTO, MAR_CERRADO, timestamp=0.0, config=CONFIG
    )
    assert eventos == []
    assert nuevo_estado.cierre_inicio is None


def test_cierre_breve_no_dispara_microsueno():
    estado = estado_inicial_somnolencia()
    for t in [0.0, 0.5, 1.0]:
        estado, eventos = procesar_somnolencia(
            estado, EAR_CERRADO, MAR_CERRADO, timestamp=t, config=CONFIG
        )
        assert eventos == []


def test_cierre_exactamente_en_el_limite_no_dispara():
    # Pasos densos (<= gap_maximo_segundos) para que el chequeo de hueco no
    # interfiera con la condicion de limite que este test quiere ejercitar.
    estado = estado_inicial_somnolencia()
    for t in [0.0, 0.5, 1.0, 1.5]:
        estado, eventos = procesar_somnolencia(
            estado, EAR_CERRADO, MAR_CERRADO, timestamp=t, config=CONFIG
        )
    assert eventos == []


def test_cierre_sostenido_dispara_microsueno():
    estado = estado_inicial_somnolencia()
    for t in [0.0, 0.5, 1.0, 1.5]:
        estado, eventos = procesar_somnolencia(
            estado, EAR_CERRADO, MAR_CERRADO, timestamp=t, config=CONFIG
        )
    estado, eventos = procesar_somnolencia(
        estado, EAR_CERRADO, MAR_CERRADO, timestamp=1.6, config=CONFIG
    )
    assert eventos == [EventoSomnolencia(tipo="microsueno", valor=1.6)]


def test_microsueno_no_re_dispara_dentro_del_cooldown():
    # Muestreo continuo (paso 0.5s, bajo gap_maximo_segundos) para simular
    # ojos cerrados de forma ininterrumpida durante todo el cooldown.
    estado = estado_inicial_somnolencia()
    eventos_microsueno = []
    t = 0.0
    while t <= 31.5:
        estado, eventos = procesar_somnolencia(
            estado, EAR_CERRADO, MAR_CERRADO, timestamp=t, config=CONFIG
        )
        eventos_microsueno += [e for e in eventos if e.tipo == "microsueno"]
        t += 0.5
    # Un unico disparo (en t=2.0); el cooldown de 30s sigue activo durante
    # el resto del tramo (expira recien en t=32.0).
    assert len(eventos_microsueno) == 1
    assert eventos_microsueno[0].valor == 2.0


def test_microsueno_re_dispara_tras_cooldown_si_sigue_cerrado():
    estado = estado_inicial_somnolencia()
    eventos_microsueno = []
    t = 0.0
    while t <= 32.5:
        estado, eventos = procesar_somnolencia(
            estado, EAR_CERRADO, MAR_CERRADO, timestamp=t, config=CONFIG
        )
        eventos_microsueno += [e for e in eventos if e.tipo == "microsueno"]
        t += 0.5
    assert len(eventos_microsueno) == 2
    assert eventos_microsueno[0].valor == 2.0
    assert eventos_microsueno[1].valor == 32.0


def test_apertura_de_ojos_reinicia_temporizador_microsueno():
    estado = estado_inicial_somnolencia()
    estado, _ = procesar_somnolencia(estado, EAR_CERRADO, MAR_CERRADO, timestamp=0.0, config=CONFIG)
    estado, _ = procesar_somnolencia(estado, EAR_ABIERTO, MAR_CERRADO, timestamp=0.5, config=CONFIG)
    assert estado.cierre_inicio is None
    estado, _ = procesar_somnolencia(estado, EAR_CERRADO, MAR_CERRADO, timestamp=0.6, config=CONFIG)
    assert estado.cierre_inicio == 0.6


def test_microsueno_no_dispara_con_valor_inflado_tras_hueco_prolongado():
    # Reproduce el hallazgo de la revision final del detector de
    # distraccion (aplicable tambien aqui): si el rostro no se detecta
    # durante un tramo largo (frames descartados por completo, sin llamar
    # a procesar_somnolencia) y luego se retoma con los ojos ya cerrados,
    # el temporizador de microsueno NO debe asumir que estuvo cerrado
    # desde antes del hueco.
    estado = estado_inicial_somnolencia()
    estado, _ = procesar_somnolencia(estado, EAR_ABIERTO, MAR_CERRADO, timestamp=0.0, config=CONFIG)
    # Hueco prolongado: sin llamadas entre t=0 y t=50 (rostro no detectado).
    estado, eventos = procesar_somnolencia(estado, EAR_CERRADO, MAR_CERRADO, timestamp=50.0, config=CONFIG)
    assert eventos == []
    assert estado.cierre_inicio == 50.0


def test_perclos_no_evalua_antes_de_completar_ventana():
    estado = estado_inicial_somnolencia()
    eventos_perclos = []
    t = 0.0
    while t < 60.0:
        estado, eventos = procesar_somnolencia(
            estado, EAR_CERRADO, MAR_CERRADO, timestamp=t, config=CONFIG
        )
        eventos_perclos += [e for e in eventos if e.tipo == "perclos"]
        t += 1.0
    assert eventos_perclos == []


def test_perclos_dispara_cuando_fraccion_cerrada_supera_umbral():
    estado = estado_inicial_somnolencia()
    eventos_perclos = []
    t = 0.0
    while t <= 65.0:
        estado, eventos = procesar_somnolencia(
            estado, EAR_CERRADO, MAR_CERRADO, timestamp=t, config=CONFIG
        )
        eventos_perclos += [e for e in eventos if e.tipo == "perclos"]
        t += 1.0
    assert len(eventos_perclos) >= 1
    assert eventos_perclos[0].valor == 1.0


def test_perclos_no_dispara_si_fraccion_bajo_umbral():
    estado = estado_inicial_somnolencia()
    eventos_perclos = []
    t = 0.0
    while t <= 65.0:
        # 1 de cada 10 muestras cerrada = 10% < 15% del umbral.
        cerrado = (int(t) % 10 == 0)
        ear = EAR_CERRADO if cerrado else EAR_ABIERTO
        estado, eventos = procesar_somnolencia(estado, ear, MAR_CERRADO, timestamp=t, config=CONFIG)
        eventos_perclos += [e for e in eventos if e.tipo == "perclos"]
        t += 1.0
    assert eventos_perclos == []


def test_muestras_antiguas_se_recortan_fuera_de_la_ventana():
    estado = estado_inicial_somnolencia()
    t = 0.0
    while t <= 65.0:
        estado, _ = procesar_somnolencia(estado, EAR_CERRADO, MAR_CERRADO, timestamp=t, config=CONFIG)
        t += 1.0
    assert all(
        m.timestamp >= t - 1.0 - CONFIG.perclos_ventana_segundos for m in estado.muestras
    )


def test_estado_inicial_no_tiene_muestras():
    estado = estado_inicial_somnolencia()
    assert estado.muestras == []
    assert estado.cierre_inicio is None
    assert estado.bostezos == []
    assert estado.boca_abierta_inicio is None


def test_microsueno_y_perclos_pueden_dispararse_en_el_mismo_llamado():
    # Construye el estado previo directamente (en vez de hacerlo evolucionar
    # con muchas llamadas) para forzar que ambos temporizadores esten listos
    # para disparar en la misma llamada. Bajo cierre continuo, microsueno
    # (periodo 30s desde su primer disparo en t=2) y perclos (periodo 30s
    # desde su primer disparo en t=60) tienen fases distintas y nunca
    # coinciden solos -- por eso se fuerza la precondicion explicitamente.
    muestras_previas = [Muestra(timestamp=float(t), cerrado=True) for t in range(0, 60)]
    estado = EstadoSomnolencia(
        muestras=muestras_previas,
        cierre_inicio=58.0,
        ultimo_disparo_microsueno=None,
        ultimo_disparo_perclos=None,
        primer_timestamp=0.0,
        ultimo_procesado=59.0,
    )
    estado, eventos = procesar_somnolencia(estado, EAR_CERRADO, MAR_CERRADO, timestamp=60.0, config=CONFIG)
    tipos = {e.tipo for e in eventos}
    assert tipos == {"microsueno", "perclos"}


def test_boca_cerrada_no_acumula_apertura():
    estado = estado_inicial_somnolencia()
    nuevo_estado, eventos = procesar_somnolencia(
        estado, EAR_ABIERTO, MAR_CERRADO, timestamp=0.0, config=CONFIG
    )
    assert eventos == []
    assert nuevo_estado.boca_abierta_inicio is None


def test_apertura_breve_no_dispara_bostezo():
    estado = estado_inicial_somnolencia()
    for t in [0.0, 0.5, 1.0]:
        estado, eventos = procesar_somnolencia(
            estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=t, config=CONFIG
        )
        assert eventos == []


def test_apertura_sostenida_dispara_bostezo():
    estado = estado_inicial_somnolencia()
    for t in [0.0, 0.5, 1.0, 1.5]:
        estado, eventos = procesar_somnolencia(
            estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=t, config=CONFIG
        )
    estado, eventos = procesar_somnolencia(
        estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=1.6, config=CONFIG
    )
    assert eventos == [EventoSomnolencia(tipo="bostezo", valor=1.6)]


def test_bostezo_no_re_dispara_dentro_del_cooldown():
    estado = estado_inicial_somnolencia()
    eventos_bostezo = []
    t = 0.0
    while t <= 31.5:
        estado, eventos = procesar_somnolencia(
            estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=t, config=CONFIG
        )
        eventos_bostezo += [e for e in eventos if e.tipo == "bostezo"]
        t += 0.5
    assert len(eventos_bostezo) == 1
    assert eventos_bostezo[0].valor == 2.0


def test_cierre_de_boca_reinicia_temporizador_bostezo():
    estado = estado_inicial_somnolencia()
    estado, _ = procesar_somnolencia(estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=0.0, config=CONFIG)
    estado, _ = procesar_somnolencia(estado, EAR_ABIERTO, MAR_CERRADO, timestamp=0.5, config=CONFIG)
    assert estado.boca_abierta_inicio is None
    estado, _ = procesar_somnolencia(estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=0.6, config=CONFIG)
    assert estado.boca_abierta_inicio == 0.6


def test_bostezo_no_dispara_con_valor_inflado_tras_hueco_prolongado():
    estado = estado_inicial_somnolencia()
    estado, _ = procesar_somnolencia(estado, EAR_ABIERTO, MAR_CERRADO, timestamp=0.0, config=CONFIG)
    estado, eventos = procesar_somnolencia(estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=50.0, config=CONFIG)
    assert eventos == []
    assert estado.boca_abierta_inicio == 50.0
```

- [ ] **Step 2: Confirmar que falla**

Run: `pytest tests/test_drowsiness_state.py -v`
Expected: FAIL — `ImportError: cannot import name 'procesar_somnolencia' from 'dsd.drowsiness_state'`

- [ ] **Step 3: Reemplazar el contenido completo de `dsd/drowsiness_state.py`**

```python
from dataclasses import dataclass, field
from typing import List, Optional

from dsd.config import ConfigSomnolencia
from dsd.sustained_timer import EstadoTemporizadorSostenido, procesar_temporizador_sostenido


@dataclass
class Muestra:
    timestamp: float
    cerrado: bool


@dataclass
class EstadoSomnolencia:
    muestras: List[Muestra] = field(default_factory=list)
    cierre_inicio: Optional[float] = None
    ultimo_disparo_microsueno: Optional[float] = None
    ultimo_disparo_perclos: Optional[float] = None
    primer_timestamp: Optional[float] = None
    ultimo_procesado: Optional[float] = None
    boca_abierta_inicio: Optional[float] = None
    ultimo_disparo_bostezo: Optional[float] = None
    bostezos: List[float] = field(default_factory=list)
    ultimo_disparo_fatiga_bostezos: Optional[float] = None


@dataclass
class EventoSomnolencia:
    tipo: str
    valor: float


def estado_inicial_somnolencia() -> EstadoSomnolencia:
    return EstadoSomnolencia()


def _calcular_perclos(muestras: List[Muestra]) -> float:
    # PERCLOS ponderado por tiempo: cada intervalo entre dos muestras
    # consecutivas se atribuye al estado (cerrado/abierto) de la muestra mas
    # antigua del par. Esto es correcto incluso si el frame rate varia (a
    # diferencia de contar muestras cerradas / muestras totales, que asume
    # implicitamente frame rate constante).
    tiempo_total = 0.0
    tiempo_cerrado = 0.0
    for anterior, siguiente in zip(muestras, muestras[1:]):
        dt = siguiente.timestamp - anterior.timestamp
        tiempo_total += dt
        if anterior.cerrado:
            tiempo_cerrado += dt
    if tiempo_total == 0.0:
        return 0.0
    return tiempo_cerrado / tiempo_total


def procesar_somnolencia(
    estado: EstadoSomnolencia,
    ear: float,
    mar: float,
    timestamp: float,
    config: ConfigSomnolencia,
) -> tuple[EstadoSomnolencia, list[EventoSomnolencia]]:
    eventos: List[EventoSomnolencia] = []
    cerrado = ear < config.ear_umbral

    # Un hueco (tiempo excesivo desde el ultimo frame procesado, p.ej.
    # porque el rostro no se detecto durante un tramo) invalida el
    # supuesto de que la condicion se mantuvo continuamente durante ese
    # tramo -- ver dsd/sustained_timer.py.
    hubo_hueco = (
        estado.ultimo_procesado is not None
        and (timestamp - estado.ultimo_procesado) > config.gap_maximo_segundos
    )

    # --- Microsueno: temporizador de cierre continuo (helper compartido) ---
    temporizador_microsueno = EstadoTemporizadorSostenido(
        inicio=estado.cierre_inicio, ultimo_disparo=estado.ultimo_disparo_microsueno
    )
    temporizador_microsueno, valor_microsueno = procesar_temporizador_sostenido(
        temporizador_microsueno,
        cerrado,
        hubo_hueco,
        timestamp,
        config.microsueno_segundos,
        config.cooldown_segundos,
    )
    if valor_microsueno is not None:
        eventos.append(EventoSomnolencia(tipo="microsueno", valor=valor_microsueno))

    # --- PERCLOS: ventana deslizante ---
    primer_timestamp = estado.primer_timestamp if estado.primer_timestamp is not None else timestamp
    muestras = [
        m for m in estado.muestras if m.timestamp >= timestamp - config.perclos_ventana_segundos
    ]
    muestras.append(Muestra(timestamp=timestamp, cerrado=cerrado))

    ultimo_disparo_perclos = estado.ultimo_disparo_perclos
    ventana_cubierta = (timestamp - primer_timestamp) >= config.perclos_ventana_segundos
    # Ademas de haber transcurrido el tiempo nominal de la ventana, exige que
    # las muestras retenidas cubran una fraccion minima real de esa ventana.
    # Sin esto, si el rostro no se detecto durante un tramo largo (frames
    # descartados por completo) y luego llegan solo un par de muestras
    # cercanas entre si, "ventana_cubierta" quedaria satisfecho por el
    # tiempo transcurrido desde el inicio de la sesion aunque los datos
    # reales sean minimos, disparando PERCLOS de forma espuria.
    cobertura_real = muestras[-1].timestamp - muestras[0].timestamp if len(muestras) >= 2 else 0.0
    cobertura_suficiente = cobertura_real >= (
        config.perclos_ventana_segundos * config.perclos_cobertura_minima
    )
    if ventana_cubierta and len(muestras) >= 2 and cobertura_suficiente:
        perclos = _calcular_perclos(muestras)
        if perclos >= config.perclos_umbral:
            en_cooldown = (
                ultimo_disparo_perclos is not None
                and (timestamp - ultimo_disparo_perclos) < config.cooldown_segundos
            )
            if not en_cooldown:
                eventos.append(EventoSomnolencia(tipo="perclos", valor=perclos))
                ultimo_disparo_perclos = timestamp

    # --- Bostezo individual: temporizador de apertura continua (mismo
    # helper compartido que microsueno) ---
    boca_abierta = mar > config.mar_umbral
    temporizador_bostezo = EstadoTemporizadorSostenido(
        inicio=estado.boca_abierta_inicio, ultimo_disparo=estado.ultimo_disparo_bostezo
    )
    temporizador_bostezo, valor_bostezo = procesar_temporizador_sostenido(
        temporizador_bostezo,
        boca_abierta,
        hubo_hueco,
        timestamp,
        config.bostezo_min_segundos,
        config.cooldown_segundos,
    )
    bostezos = list(estado.bostezos)
    if valor_bostezo is not None:
        eventos.append(EventoSomnolencia(tipo="bostezo", valor=valor_bostezo))
        bostezos.append(timestamp)

    nuevo_estado = EstadoSomnolencia(
        muestras=muestras,
        cierre_inicio=temporizador_microsueno.inicio,
        ultimo_disparo_microsueno=temporizador_microsueno.ultimo_disparo,
        ultimo_disparo_perclos=ultimo_disparo_perclos,
        primer_timestamp=primer_timestamp,
        ultimo_procesado=timestamp,
        boca_abierta_inicio=temporizador_bostezo.inicio,
        ultimo_disparo_bostezo=temporizador_bostezo.ultimo_disparo,
        bostezos=bostezos,
        ultimo_disparo_fatiga_bostezos=estado.ultimo_disparo_fatiga_bostezos,
    )
    return nuevo_estado, eventos
```

- [ ] **Step 4: Confirmar que pasa**

Run: `pytest tests/test_drowsiness_state.py -v`
Expected: todos los tests PASS

- [ ] **Step 5: Confirmar que el resto de la suite sigue pasando**

Run: `pytest -v`
Expected: todos PASS (nada más en el repo llama a `procesar_ear`; `dsd/main.py` se actualiza recién en Task 6, así que quedará con una llamada rota a `procesar_ear` hasta entonces — es esperado, `main.py` no tiene test automatizado que lo ejecute).

- [ ] **Step 6: Commit**

```bash
git add dsd/drowsiness_state.py tests/test_drowsiness_state.py
git commit -m "feat: renombrar procesar_ear a procesar_somnolencia y agregar deteccion de bostezo individual"
```

---

## Task 5: `dsd/drowsiness_state.py` — fatiga por frecuencia de bostezos

**Files:**
- Modify: `dsd/drowsiness_state.py`
- Modify: `tests/test_drowsiness_state.py`

**Interfaces:**
- Consumes: `estado.bostezos` y `estado.ultimo_disparo_fatiga_bostezos` ya agregados en Task 4.
- Produces: `procesar_somnolencia` ahora también puede devolver `EventoSomnolencia(tipo="fatiga_bostezos", valor=<cantidad>)`.

- [ ] **Step 1: Escribir el test que falla**

Agregar al final de `tests/test_drowsiness_state.py`:

```python


def test_fatiga_bostezos_no_dispara_antes_de_alcanzar_la_cantidad():
    estado = estado_inicial_somnolencia()
    eventos_fatiga = []
    for inicio in [0.0, 40.0]:
        for t in [inicio, inicio + 0.5, inicio + 1.0, inicio + 1.5, inicio + 1.6]:
            estado, eventos = procesar_somnolencia(
                estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=t, config=CONFIG
            )
            eventos_fatiga += [e for e in eventos if e.tipo == "fatiga_bostezos"]
    assert eventos_fatiga == []
    assert len(estado.bostezos) == 2


def test_fatiga_bostezos_dispara_al_alcanzar_la_cantidad_en_la_ventana():
    estado = estado_inicial_somnolencia()
    eventos_fatiga = []
    for inicio in [0.0, 40.0, 80.0]:
        for t in [inicio, inicio + 0.5, inicio + 1.0, inicio + 1.5, inicio + 1.6]:
            estado, eventos = procesar_somnolencia(
                estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=t, config=CONFIG
            )
            eventos_fatiga += [e for e in eventos if e.tipo == "fatiga_bostezos"]
    assert len(eventos_fatiga) == 1
    assert eventos_fatiga[0].valor == 3.0


def test_bostezos_fuera_de_la_ventana_se_recortan():
    estado = estado_inicial_somnolencia()
    for inicio in [0.0, 40.0, 80.0]:
        for t in [inicio, inicio + 0.5, inicio + 1.0, inicio + 1.5, inicio + 1.6]:
            estado, _ = procesar_somnolencia(
                estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=t, config=CONFIG
            )
    # Cuarto bostezo bien fuera de la ventana de 300s respecto al primero:
    # solo deberian quedar los bostezos dentro de los ultimos 300s.
    inicio = 350.0
    for t in [inicio, inicio + 0.5, inicio + 1.0, inicio + 1.5, inicio + 1.6]:
        estado, _ = procesar_somnolencia(
            estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=t, config=CONFIG
        )
    assert all(
        ts >= (inicio + 1.6) - CONFIG.bostezo_ventana_segundos for ts in estado.bostezos
    )


def test_fatiga_bostezos_no_re_dispara_dentro_del_cooldown():
    estado = estado_inicial_somnolencia()
    eventos_fatiga = []
    for inicio in [0.0, 40.0, 80.0]:
        for t in [inicio, inicio + 0.5, inicio + 1.0, inicio + 1.5, inicio + 1.6]:
            estado, eventos = procesar_somnolencia(
                estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=t, config=CONFIG
            )
            eventos_fatiga += [e for e in eventos if e.tipo == "fatiga_bostezos"]
    # Tercer bostezo (t=81.6) hace que la cuenta llegue a 3 y dispara
    # fatiga_bostezos. Los saltos de tiempo siguientes (>gap_maximo_segundos)
    # representan tramos con la boca cerrada -- no deberian generar nuevos
    # bostezos, y mientras la cuenta en ventana siga en 3, fatiga_bostezos
    # no debe volver a dispararse antes de que expire su propio cooldown.
    for t in [90.0, 100.0, 109.9]:
        estado, eventos = procesar_somnolencia(
            estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=t, config=CONFIG
        )
        eventos_fatiga += [e for e in eventos if e.tipo == "fatiga_bostezos"]
    assert len(eventos_fatiga) == 1


def test_fatiga_bostezos_re_dispara_tras_cooldown_si_la_cuenta_sigue_alta():
    estado = estado_inicial_somnolencia()
    eventos_fatiga = []
    for inicio in [0.0, 40.0, 80.0]:
        for t in [inicio, inicio + 0.5, inicio + 1.0, inicio + 1.5, inicio + 1.6]:
            estado, eventos = procesar_somnolencia(
                estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=t, config=CONFIG
            )
            eventos_fatiga += [e for e in eventos if e.tipo == "fatiga_bostezos"]
    # Cooldown de fatiga_bostezos expira en t=81.6+30=111.6; a t=112.0 el
    # conteo de bostezos en ventana sigue en 3, asi que vuelve a disparar.
    estado, eventos = procesar_somnolencia(
        estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=112.0, config=CONFIG
    )
    eventos_fatiga += [e for e in eventos if e.tipo == "fatiga_bostezos"]
    assert len(eventos_fatiga) == 2
```

- [ ] **Step 2: Confirmar que falla**

Run: `pytest tests/test_drowsiness_state.py -v`
Expected: FAIL en los 5 tests nuevos — `assert [] == []` pasa por casualidad en el primero, pero `test_fatiga_bostezos_dispara_al_alcanzar_la_cantidad_en_la_ventana` falla con `assert 0 == 1` (ningún evento `fatiga_bostezos` se genera todavía).

- [ ] **Step 3: Agregar la lógica de ventana de frecuencia**

En `dsd/drowsiness_state.py`, reemplazar el bloque `# --- Bostezo individual ... ---` y la construcción de `nuevo_estado` (todo lo que sigue a la sección de PERCLOS) por:

```python
    # --- Bostezo individual: temporizador de apertura continua (mismo
    # helper compartido que microsueno) ---
    boca_abierta = mar > config.mar_umbral
    temporizador_bostezo = EstadoTemporizadorSostenido(
        inicio=estado.boca_abierta_inicio, ultimo_disparo=estado.ultimo_disparo_bostezo
    )
    temporizador_bostezo, valor_bostezo = procesar_temporizador_sostenido(
        temporizador_bostezo,
        boca_abierta,
        hubo_hueco,
        timestamp,
        config.bostezo_min_segundos,
        config.cooldown_segundos,
    )
    bostezos = list(estado.bostezos)
    if valor_bostezo is not None:
        eventos.append(EventoSomnolencia(tipo="bostezo", valor=valor_bostezo))
        bostezos.append(timestamp)

    # --- Fatiga por frecuencia de bostezos: ventana deslizante (mismo
    # principio que PERCLOS, pero contando ocurrencias discretas en vez de
    # tiempo ponderado -- la frecuencia de bostezos, no la duracion de cada
    # uno, es el indicador de fatiga acumulada). ---
    bostezos = [t for t in bostezos if t >= timestamp - config.bostezo_ventana_segundos]
    ultimo_disparo_fatiga_bostezos = estado.ultimo_disparo_fatiga_bostezos
    if len(bostezos) >= config.bostezo_umbral_cantidad:
        en_cooldown = (
            ultimo_disparo_fatiga_bostezos is not None
            and (timestamp - ultimo_disparo_fatiga_bostezos) < config.cooldown_segundos
        )
        if not en_cooldown:
            eventos.append(EventoSomnolencia(tipo="fatiga_bostezos", valor=float(len(bostezos))))
            ultimo_disparo_fatiga_bostezos = timestamp

    nuevo_estado = EstadoSomnolencia(
        muestras=muestras,
        cierre_inicio=temporizador_microsueno.inicio,
        ultimo_disparo_microsueno=temporizador_microsueno.ultimo_disparo,
        ultimo_disparo_perclos=ultimo_disparo_perclos,
        primer_timestamp=primer_timestamp,
        ultimo_procesado=timestamp,
        boca_abierta_inicio=temporizador_bostezo.inicio,
        ultimo_disparo_bostezo=temporizador_bostezo.ultimo_disparo,
        bostezos=bostezos,
        ultimo_disparo_fatiga_bostezos=ultimo_disparo_fatiga_bostezos,
    )
    return nuevo_estado, eventos
```

- [ ] **Step 4: Confirmar que pasa**

Run: `pytest tests/test_drowsiness_state.py -v`
Expected: todos los tests PASS (incluye los 5 nuevos)

- [ ] **Step 5: Confirmar que el resto de la suite sigue pasando**

Run: `pytest -v`
Expected: todos PASS

- [ ] **Step 6: Commit**

```bash
git add dsd/drowsiness_state.py tests/test_drowsiness_state.py
git commit -m "feat: agregar deteccion de fatiga por frecuencia de bostezos"
```

---

## Task 6: `dsd/main.py` — integración

**Files:**
- Modify: `dsd/main.py`

**Interfaces:**
- Consumes: `calcular_mar` (Task 1), `procesar_somnolencia` (Tasks 4-5), `ResultadoLandmarks.puntos_boca` (Task 2), `ConfigSomnolencia` con los campos nuevos (Task 3).

No hay test automatizado para `dsd/main.py` (mismo criterio que el resto del archivo: requiere cámara real). Verificación manual en Step 4.

- [ ] **Step 1: Actualizar los imports**

En `dsd/main.py`, reemplazar:

```python
from dsd.drowsiness_state import estado_inicial_somnolencia, procesar_ear
from dsd.eye_metrics import calcular_ear
from dsd.face_mesh import detectar_landmarks
from dsd.gaze_metrics import calcular_gaze_ratio
```

por:

```python
from dsd.drowsiness_state import estado_inicial_somnolencia, procesar_somnolencia
from dsd.eye_metrics import calcular_ear
from dsd.face_mesh import detectar_landmarks
from dsd.gaze_metrics import calcular_gaze_ratio
from dsd.mouth_metrics import calcular_mar
```

- [ ] **Step 2: Calcular MAR y llamar a `procesar_somnolencia`**

Reemplazar:

```python
                    ear_derecho = calcular_ear(landmarks.puntos_ojo_derecho)
                    ear_izquierdo = calcular_ear(landmarks.puntos_ojo_izquierdo)
                    ear_promedio = (ear_derecho + ear_izquierdo) / 2
                    estado_somnolencia, eventos_somnolencia = procesar_ear(
                        estado_somnolencia, ear_promedio, timestamp, config_somnolencia
                    )
```

por:

```python
                    ear_derecho = calcular_ear(landmarks.puntos_ojo_derecho)
                    ear_izquierdo = calcular_ear(landmarks.puntos_ojo_izquierdo)
                    ear_promedio = (ear_derecho + ear_izquierdo) / 2
                    mar = calcular_mar(landmarks.puntos_boca)
                    estado_somnolencia, eventos_somnolencia = procesar_somnolencia(
                        estado_somnolencia, ear_promedio, mar, timestamp, config_somnolencia
                    )
```

El resto del bloque (impresión en consola + `registrar_evento` para cada `evento_somnolencia`) no cambia — ya usa `evento_somnolencia.tipo`/`.valor` de forma genérica, así que cubre `bostezo` y `fatiga_bostezos` sin modificaciones.

- [ ] **Step 3: Confirmar que la suite completa sigue pasando**

Run: `pytest -v`
Expected: todos los tests PASS

- [ ] **Step 4: Verificación manual (cámara real)**

```bash
source .venv/bin/activate
python -m dsd.main
```

Con una sesión activa (conductor ya enrolado y reconocido):
1. Bostezar (boca bien abierta) de forma sostenida por más de 1.5s → debe imprimirse `Evento de somnolencia: bostezo (valor=X.XXX)` en la consola.
2. Repetir el bostezo 3 veces dentro de una ventana de 5 minutos (respetando los ~30s de cooldown entre bostezos individuales) → al tercer bostezo debe imprimirse además `Evento de somnolencia: fatiga_bostezos (valor=3.000)`.
3. Salir con `q` y verificar en `data/app.db` que los eventos quedaron persistidos:

```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/app.db')
for row in conn.execute(\"SELECT tipo, valor, timestamp FROM events WHERE tipo IN ('bostezo', 'fatiga_bostezos') ORDER BY id DESC LIMIT 10\"):
    print(row)
"
```
Expected: filas con `tipo='bostezo'` y (si se completaron los 3 bostezos) `tipo='fatiga_bostezos'`.

- [ ] **Step 5: Commit**

```bash
git add dsd/main.py
git commit -m "feat: integrar deteccion de bostezo en el loop principal"
```

---

## Self-Review

**Cobertura de la spec:**
- MAR (idéntico geométricamente al EAR) → Task 1. ✓
- Índices de boca verificados manualmente → Task 2 (constantes documentadas, ya verificadas en la sesión de brainstorming). ✓
- `mar_umbral`, `bostezo_min_segundos`, `bostezo_ventana_segundos`, `bostezo_umbral_cantidad` en config → Task 3. ✓
- `procesar_ear` → `procesar_somnolencia`, reutilizando `sustained_timer` → Task 4. ✓
- Evento `bostezo` individual con cooldown reutilizado → Task 4. ✓
- Evento `fatiga_bostezos` por ventana deslizante, estilo PERCLOS → Task 5. ✓
- Reinicio tras hueco de detección (ambos temporizadores) → cubierto por `hubo_hueco` ya compartido, testeado en `test_bostezo_no_dispara_con_valor_inflado_tras_hueco_prolongado` (Task 4). ✓
- Persistencia sin cambios de esquema en `dsd/db.py` → confirmado, ningún task lo toca. ✓
- Integración en `dsd/main.py` con verificación manual → Task 6. ✓
- Fuera de alcance (alertas, calibración final de umbrales, celular/cigarro, Orange Pi) → ningún task los incluye. ✓

Sin gaps encontrados.
