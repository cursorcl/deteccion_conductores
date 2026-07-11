# Detector de Distracción (pose de cabeza + mirada) — Plan de Implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detectar distracción del conductor combinando pose de cabeza (yaw/pitch, vía la matriz de rotación de Mediapipe) y dirección de mirada (ratio de posición del iris), activo solo mientras `dsd.session_state` reporta `Estado.ACTIVA`, persistiendo eventos `distraccion_cabeza` y `distraccion_mirada` en la tabla `events` ya existente. Solo detección — sin alertas (sub-proyecto futuro, igual que somnolencia).

**Architecture:** Se ejecuta en el hilo principal (igual que somnolencia), sobre cada frame, dentro del mismo bloque `Estado.ACTIVA`. `dsd/face_mesh.py` se modifica para correr `FaceLandmarker.detect()` **una sola vez** por frame (en vez de una vez por detector) y devolver toda la información necesaria (puntos de ojos para EAR, puntos de iris, matriz de rotación). Dos módulos puros nuevos (`dsd/head_pose.py`, `dsd/gaze_metrics.py`) convierten esos datos crudos en señales interpretables (grados, ratios); un tercer módulo puro (`dsd/distraction_state.py`) aplica el mismo patrón de temporizador-sostenido + cooldown ya usado para microsueño en somnolencia, mostrando por qué NO se necesita una ventana acumulada tipo PERCLOS aquí (la literatura de distracción se basa en la duración de un solo vistazo, no en fracción acumulada).

**Tech Stack:** Python 3.11.11, Mediapipe Face Landmarker (Tasks API, ya en uso), PyYAML, `sqlite3` (stdlib), `pytest`. **Sin dependencias nuevas** — `mediapipe==0.10.35`, `PyYAML==6.0.3` y `numpy==2.4.6` ya están en `requirements.txt` desde el sub-proyecto de somnolencia.

## Global Constraints

- **Archivo de configuración:** `config/distraccion.yaml`, cargado con `dsd.config.cargar_config_distraccion(path: str) -> ConfigDistraccion`. Valores exactos:
  - `distraccion_segundos: 2.0`
  - `yaw_umbral_grados: 20`
  - `pitch_umbral_grados: 15`
  - `gaze_ratio_umbral: 0.20`
  - `cooldown_segundos: 30`
- **Tabla `events`: sin cambios de esquema.** Ya es genérica (`session_id, tipo, valor, timestamp, synced`). Los nuevos eventos usan `tipo="distraccion_cabeza"` o `tipo="distraccion_mirada"` con el `registrar_evento` ya existente en `dsd/db.py`. **No se toca `dsd/db.py` en este plan.**
- **Semántica de `valor`:** para ambos tipos, la duración en segundos del vistazo/giro sostenido al momento del disparo (mismo significado que `valor` de `microsueno` en somnolencia).
- **Semántica de cooldown:** por tipo de evento (`distraccion_cabeza` y `distraccion_mirada` tienen cooldowns independientes), wall-clock desde el último disparo de ese tipo — mismo patrón que somnolencia.
- **Reinicio de estado:** `dsd/main.py` crea un `EstadoDistraccion` nuevo (`estado_inicial_distraccion()`) en el mismo momento en que reinicia `EstadoSomnolencia`, al procesar el evento `sesion_iniciada`.
- **Una sola inferencia de Mediapipe por frame:** `dsd/face_mesh.py` expone `detectar_landmarks(frame) -> Optional[ResultadoLandmarks]`, que reemplaza a `detectar_ojos`. Los 6 puntos de ojo por lado (mismos índices `INDICES_OJO_DERECHO`/`INDICES_OJO_IZQUIERDO` ya usados para EAR) sirven también como referencia geométrica para el ratio de mirada — no se necesitan índices adicionales para eso.
- **Índices de iris** (topología de 478 puntos: 468 base + 10 de iris): centro del iris derecho = índice `468`, centro del iris izquierdo = índice `473`. **Nota:** esta asignación se verifica manualmente en la Tarea 5 (ver contingencia en esa tarea) antes de darla por buena, igual que se hizo con los índices de ojos en el sub-proyecto anterior.
- **Matriz de rotación:** se activa `output_facial_transformation_matrixes=True` en `FaceLandmarkerOptions` (hoy en `False`). Se usa solo la submatriz de rotación 3x3 (esquina superior izquierda de la matriz 4x4 homogénea que devuelve Mediapipe).
- **Sin ventana acumulada tipo PERCLOS:** a diferencia de somnolencia, `distraction_state.py` NO necesita un mecanismo de ventana deslizante ni cobertura mínima — la literatura de distracción (ver Tarea 3) se basa en la duración de un único vistazo continuo, no en una fracción de tiempo acumulada.
- **Frames sin rostro detectado durante `ACTIVA`:** se descartan por completo (no se llama a `procesar_pose_y_mirada`), mismo principio que somnolencia.
- **Fuera de alcance:** alertas audio/visuales, sincronización con la central, detectores de celular/cigarro, puerto a Orange Pi.

---

### Task 1: `dsd/head_pose.py` — extracción pura de yaw/pitch desde una matriz de rotación

**Files:**
- Create: `dsd/head_pose.py`
- Test: `tests/test_head_pose.py`

**Interfaces:**
- Consumes: nada (módulo puro, sin dependencias de otros módulos del proyecto).
- Produces: `calcular_yaw_pitch(matriz_rotacion: Sequence[Sequence[float]]) -> Tuple[float, float]` — usado por `dsd/main.py` en la Tarea 6.

- [ ] **Step 1: Escribir los tests que deben fallar**

`tests/test_head_pose.py`:
```python
import math

import pytest

from dsd.head_pose import calcular_yaw_pitch


def _matriz_rotacion(yaw_grados: float, pitch_grados: float, roll_grados: float = 0.0):
    """Construye una matriz de rotacion 3x3 sintetica R = Ry(yaw) @ Rx(pitch)
    @ Rz(roll), usada solo para generar datos de prueba (no depende de
    Mediapipe ni de camara real)."""
    yaw = math.radians(yaw_grados)
    pitch = math.radians(pitch_grados)
    roll = math.radians(roll_grados)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    return [
        [cy * cr + sy * sp * sr, -cy * sr + sy * sp * cr, sy * cp],
        [cp * sr, cp * cr, -sp],
        [-sy * cr + cy * sp * sr, sy * sr + cy * sp * cr, cy * cp],
    ]


def test_matriz_identidad_da_yaw_pitch_cero():
    yaw, pitch = calcular_yaw_pitch(_matriz_rotacion(0.0, 0.0))
    assert yaw == pytest.approx(0.0, abs=1e-9)
    assert pitch == pytest.approx(0.0, abs=1e-9)


def test_yaw_positivo_se_recupera_correctamente():
    yaw, pitch = calcular_yaw_pitch(_matriz_rotacion(30.0, 0.0))
    assert yaw == pytest.approx(30.0)
    assert pitch == pytest.approx(0.0, abs=1e-9)


def test_yaw_negativo_se_recupera_correctamente():
    yaw, pitch = calcular_yaw_pitch(_matriz_rotacion(-25.0, 0.0))
    assert yaw == pytest.approx(-25.0)
    assert pitch == pytest.approx(0.0, abs=1e-9)


def test_pitch_positivo_se_recupera_correctamente():
    yaw, pitch = calcular_yaw_pitch(_matriz_rotacion(0.0, 20.0))
    assert yaw == pytest.approx(0.0, abs=1e-9)
    assert pitch == pytest.approx(20.0)


def test_pitch_negativo_se_recupera_correctamente():
    yaw, pitch = calcular_yaw_pitch(_matriz_rotacion(0.0, -15.0))
    assert yaw == pytest.approx(0.0, abs=1e-9)
    assert pitch == pytest.approx(-15.0)


def test_yaw_y_pitch_combinados_se_recuperan_correctamente():
    yaw, pitch = calcular_yaw_pitch(_matriz_rotacion(15.0, 10.0))
    assert yaw == pytest.approx(15.0)
    assert pitch == pytest.approx(10.0)


def test_roll_no_afecta_yaw_ni_pitch_calculados():
    yaw, pitch = calcular_yaw_pitch(_matriz_rotacion(15.0, 10.0, roll_grados=45.0))
    assert yaw == pytest.approx(15.0)
    assert pitch == pytest.approx(10.0)
```

- [ ] **Step 2: Ejecutar los tests y confirmar que fallan**

Run: `source .venv/bin/activate && pytest tests/test_head_pose.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dsd.head_pose'`

- [ ] **Step 3: Implementar `dsd/head_pose.py`**

```python
import math
from typing import Sequence, Tuple


def calcular_yaw_pitch(matriz_rotacion: Sequence[Sequence[float]]) -> Tuple[float, float]:
    """Extrae (yaw, pitch), en grados, desde una matriz de rotacion 3x3.

    Asume la convencion R = Ry(yaw) @ Rx(pitch) @ Rz(roll) (angulos de
    Euler intrinsecos): yaw es la rotacion horizontal (eje Y, girar la
    cabeza hacia un costado); pitch es la rotacion vertical (eje X,
    inclinar la cabeza hacia arriba/abajo). roll (inclinacion lateral,
    eje Z) no se calcula porque no se usa para detectar distraccion.

    Formula de extraccion (valida para |pitch| < 90 grados, rango muy
    por encima de lo que ocurre al conducir):
        pitch = asin(-R[1][2])
        yaw   = atan2(R[0][2], R[2][2])
    """
    r02 = matriz_rotacion[0][2]
    r12 = matriz_rotacion[1][2]
    r22 = matriz_rotacion[2][2]

    seno_pitch = max(-1.0, min(1.0, -r12))
    pitch_rad = math.asin(seno_pitch)
    yaw_rad = math.atan2(r02, r22)

    return math.degrees(yaw_rad), math.degrees(pitch_rad)
```

- [ ] **Step 4: Ejecutar los tests y confirmar que pasan**

Run: `pytest tests/test_head_pose.py -v`
Expected: PASS — 7 tests verdes.

- [ ] **Step 5: Commit**

```bash
git add dsd/head_pose.py tests/test_head_pose.py
git commit -m "feat: extraccion pura de yaw/pitch desde matriz de rotacion"
```

---

### Task 2: `dsd/gaze_metrics.py` — cálculo puro del ratio de mirada

**Files:**
- Create: `dsd/gaze_metrics.py`
- Test: `tests/test_gaze_metrics.py`

**Interfaces:**
- Consumes: nada (módulo puro).
- Produces: `calcular_gaze_ratio(iris: Tuple[float, float], puntos_ojo: Sequence[Tuple[float, float]]) -> Tuple[float, float]` — usado por `dsd/main.py` en la Tarea 6.

- [ ] **Step 1: Escribir los tests que deben fallar**

`tests/test_gaze_metrics.py`:
```python
import pytest

from dsd.gaze_metrics import calcular_gaze_ratio

# Mismo orden de 6 puntos EAR: (esquina_externa, parpado_superior_1,
# parpado_superior_2, esquina_interna, parpado_inferior_2, parpado_inferior_1)
PUNTOS_OJO = [
    (0.0, 0.0), (0.3, -0.15), (0.7, -0.15), (1.0, 0.0), (0.7, 0.15), (0.3, 0.15)
]


def test_iris_centrado_da_ratios_0_5():
    ratio_h, ratio_v = calcular_gaze_ratio((0.5, 0.0), PUNTOS_OJO)
    assert ratio_h == pytest.approx(0.5)
    assert ratio_v == pytest.approx(0.5)


def test_iris_cerca_de_esquina_externa_da_ratio_horizontal_bajo():
    ratio_h, _ = calcular_gaze_ratio((0.05, 0.0), PUNTOS_OJO)
    assert ratio_h == pytest.approx(0.05)


def test_iris_cerca_de_esquina_interna_da_ratio_horizontal_alto():
    ratio_h, _ = calcular_gaze_ratio((0.95, 0.0), PUNTOS_OJO)
    assert ratio_h == pytest.approx(0.95)


def test_iris_cerca_del_parpado_superior_da_ratio_vertical_bajo():
    _, ratio_v = calcular_gaze_ratio((0.5, -0.14), PUNTOS_OJO)
    assert ratio_v == pytest.approx(0.01 / 0.3)


def test_iris_cerca_del_parpado_inferior_da_ratio_vertical_alto():
    _, ratio_v = calcular_gaze_ratio((0.5, 0.14), PUNTOS_OJO)
    assert ratio_v == pytest.approx(0.29 / 0.3)


def test_calcular_gaze_ratio_lanza_error_si_no_son_6_puntos():
    with pytest.raises(ValueError):
        calcular_gaze_ratio((0.5, 0.0), [(0.0, 0.0), (1.0, 1.0)])


def test_ancho_cero_retorna_ratio_horizontal_centrado():
    puntos_degenerados = [
        (0.5, 0.0), (0.5, -0.15), (0.5, -0.15), (0.5, 0.0), (0.5, 0.15), (0.5, 0.15)
    ]
    ratio_h, _ = calcular_gaze_ratio((0.5, 0.0), puntos_degenerados)
    assert ratio_h == 0.5
```

- [ ] **Step 2: Ejecutar los tests y confirmar que fallan**

Run: `pytest tests/test_gaze_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dsd.gaze_metrics'`

- [ ] **Step 3: Implementar `dsd/gaze_metrics.py`**

```python
from typing import Sequence, Tuple


def calcular_gaze_ratio(
    iris: Tuple[float, float], puntos_ojo: Sequence[Tuple[float, float]]
) -> Tuple[float, float]:
    """Calcula la posicion relativa del iris dentro del contorno del ojo.

    `puntos_ojo` debe tener el mismo orden de 6 puntos usado para EAR:
    (esquina_externa, parpado_superior_1, parpado_superior_2,
    esquina_interna, parpado_inferior_2, parpado_inferior_1).

    Retorna (ratio_horizontal, ratio_vertical): 0.5 en cualquiera de los
    dos significa iris centrado en esa dimension; valores que se alejan
    de 0.5 indican mirada desviada hacia una esquina/parpado. El eje
    horizontal va de esquina_externa (0.0) a esquina_interna (1.0); el
    eje vertical va de parpado_superior (0.0) a parpado_inferior (1.0).
    """
    if len(puntos_ojo) != 6:
        raise ValueError(
            "calcular_gaze_ratio requiere exactamente 6 puntos de ojo "
            f"(mismo orden que EAR); se recibieron {len(puntos_ojo)}."
        )

    esquina_externa, sup1, sup2, esquina_interna, inf2, inf1 = puntos_ojo
    iris_x, iris_y = iris

    ancho = esquina_interna[0] - esquina_externa[0]
    ratio_horizontal = (iris_x - esquina_externa[0]) / ancho if ancho != 0.0 else 0.5

    y_superior = (sup1[1] + sup2[1]) / 2
    y_inferior = (inf1[1] + inf2[1]) / 2
    alto = y_inferior - y_superior
    ratio_vertical = (iris_y - y_superior) / alto if alto != 0.0 else 0.5

    return ratio_horizontal, ratio_vertical
```

- [ ] **Step 4: Ejecutar los tests y confirmar que pasan**

Run: `pytest tests/test_gaze_metrics.py -v`
Expected: PASS — 7 tests verdes.

- [ ] **Step 5: Commit**

```bash
git add dsd/gaze_metrics.py tests/test_gaze_metrics.py
git commit -m "feat: calculo puro del ratio de posicion del iris (gaze)"
```

---

### Task 3: `dsd/config.py` (MODIFY) — `ConfigDistraccion` y `config/distraccion.yaml`

**Files:**
- Modify: `dsd/config.py`
- Modify: `tests/test_config.py`
- Create: `config/distraccion.yaml`

**Interfaces:**
- Consumes: nada nuevo.
- Produces:
  - `class ConfigDistraccion` (dataclass) con campos `distraccion_segundos: float`, `yaw_umbral_grados: float`, `pitch_umbral_grados: float`, `gaze_ratio_umbral: float`, `cooldown_segundos: float`.
  - `cargar_config_distraccion(path: str) -> ConfigDistraccion`

- [ ] **Step 1: Agregar los tests que deben fallar (agregar al final de `tests/test_config.py`, sin tocar los tests de somnolencia ya existentes)**

Agregar a `tests/test_config.py`:
```python
from dsd.config import ConfigDistraccion, cargar_config_distraccion

YAML_VALIDO_DISTRACCION = """
distraccion_segundos: 2.0
yaw_umbral_grados: 20
pitch_umbral_grados: 15
gaze_ratio_umbral: 0.20
cooldown_segundos: 30
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
```

La línea de import ya existente en `tests/test_config.py` (`from dsd.config import ConfigSomnolencia, cargar_config`) debe quedar como una línea separada adicional (no reemplazar la existente), es decir el archivo termina con ambas líneas de import al principio:
```python
import pytest

from dsd.config import ConfigSomnolencia, cargar_config
from dsd.config import ConfigDistraccion, cargar_config_distraccion
```

- [ ] **Step 2: Ejecutar los tests y confirmar que fallan**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ImportError: cannot import name 'ConfigDistraccion' from 'dsd.config'`

- [ ] **Step 3: Agregar a `dsd/config.py` (al final del archivo, sin tocar `ConfigSomnolencia`/`cargar_config` ya existentes)**

Agregar a `dsd/config.py`:
```python
CAMPOS_REQUERIDOS_DISTRACCION = (
    "distraccion_segundos",
    "yaw_umbral_grados",
    "pitch_umbral_grados",
    "gaze_ratio_umbral",
    "cooldown_segundos",
)


@dataclass
class ConfigDistraccion:
    distraccion_segundos: float
    yaw_umbral_grados: float
    pitch_umbral_grados: float
    gaze_ratio_umbral: float
    cooldown_segundos: float


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
    )
```

- [ ] **Step 4: Crear `config/distraccion.yaml`**

`config/distraccion.yaml`:
```yaml
# Configuracion de deteccion de distraccion (pose de cabeza + mirada).
# Los indices de landmarks de ojos e iris usados estan documentados como
# constantes en dsd/face_mesh.py.

# Duracion minima continua (segundos) con la cabeza girada o la mirada
# desviada del frente para considerar "distraccion". Basado en Klauer et
# al., 2006 ("The Impact of Driver Inattention on Near-Crash/Crash Risk",
# NHTSA/VTTI 100-Car Naturalistic Driving Study): apartar la vista del
# camino por mas de 2 segundos aumenta sustancialmente el riesgo de
# choque/casi-choque. Se usa el mismo valor para ambos tipos de evento.
distraccion_segundos: 2.0

# Angulo de giro horizontal de la cabeza (grados) mas alla del cual se
# considera que el conductor mira hacia un costado (radio, pasajero,
# espejo lateral). No hay un unico estudio con un numero universal exacto;
# el valor esta informado por la practica comun de sistemas DMS (Driver
# Monitoring Systems) automotrices y el protocolo de evaluacion de DMS de
# Euro NCAP, que usan angulos de esta magnitud como referencia de "mirada
# fuera de la carretera". Decision de ingenieria, sujeta a calibracion con
# pruebas reales.
yaw_umbral_grados: 20

# Angulo de inclinacion vertical de la cabeza (grados) mas alla del cual
# se considera que el conductor mira hacia abajo (p.ej. el celular en el
# regazo). Misma fuente/naturaleza que yaw_umbral_grados.
pitch_umbral_grados: 15

# Desviacion minima (0.0-1.0) del ratio de mirada respecto al centro (0.5)
# para considerar que los ojos se desviaron del frente, incluso sin que la
# cabeza gire. Decision de ingenieria (no hay un estudio que fije este
# numero exacto en terminos de razon geometrica) -- se calibrara con
# verificacion manual real, igual que se hizo con ear_umbral en el
# detector de somnolencia.
gaze_ratio_umbral: 0.20

# Tiempo minimo (segundos) entre dos disparos consecutivos del mismo tipo
# de evento. Mismo valor y misma justificacion que cooldown_segundos en
# somnolencia.yaml, por consistencia entre detectores.
cooldown_segundos: 30
```

- [ ] **Step 5: Ejecutar los tests y confirmar que pasan**

Run: `pytest tests/test_config.py -v`
Expected: PASS — 8 tests verdes (4 de somnolencia ya existentes + 4 nuevos de distracción).

- [ ] **Step 6: Commit**

```bash
git add dsd/config.py config/distraccion.yaml tests/test_config.py
git commit -m "feat: carga de configuracion YAML de umbrales de distraccion"
```

---

### Task 4: `dsd/distraction_state.py` — máquina de estados pura de distracción

**Files:**
- Create: `dsd/distraction_state.py`
- Test: `tests/test_distraction_state.py`

**Interfaces:**
- Consumes: `ConfigDistraccion` de `dsd.config` (Task 3) — solo como parámetro de tipo, sin importar `cargar_config_distraccion`.
- Produces:
  - `class EstadoDistraccion` (dataclass)
  - `class EventoDistraccion` (dataclass) con campos `tipo: str`, `valor: float`
  - `estado_inicial_distraccion() -> EstadoDistraccion`
  - `procesar_pose_y_mirada(estado: EstadoDistraccion, yaw: float, pitch: float, gaze_horizontal: float, gaze_vertical: float, timestamp: float, config: ConfigDistraccion) -> tuple[EstadoDistraccion, list[EventoDistraccion]]`

- [ ] **Step 1: Escribir los tests que deben fallar**

`tests/test_distraction_state.py`:
```python
from dsd.config import ConfigDistraccion
from dsd.distraction_state import (
    EventoDistraccion,
    estado_inicial_distraccion,
    procesar_pose_y_mirada,
)

CONFIG = ConfigDistraccion(
    distraccion_segundos=2.0,
    yaw_umbral_grados=20.0,
    pitch_umbral_grados=15.0,
    gaze_ratio_umbral=0.20,
    cooldown_segundos=30.0,
)

# Valores de conveniencia: "al frente" no dispara ninguna senal; "girada"
# supera el umbral de yaw; "desviada" supera el umbral de gaze horizontal.
YAW_FRENTE = 0.0
YAW_GIRADA = 25.0
PITCH_FRENTE = 0.0
GAZE_CENTRADO = 0.5
GAZE_DESVIADO = 0.75


def test_estado_inicial_no_tiene_temporizadores_activos():
    estado = estado_inicial_distraccion()
    assert estado.cabeza_girada_inicio is None
    assert estado.mirada_desviada_inicio is None
    assert estado.ultimo_disparo_cabeza is None
    assert estado.ultimo_disparo_mirada is None


def test_frente_sin_giro_ni_desviacion_no_dispara_nada():
    estado = estado_inicial_distraccion()
    nuevo_estado, eventos = procesar_pose_y_mirada(
        estado, YAW_FRENTE, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
        timestamp=0.0, config=CONFIG,
    )
    assert eventos == []
    assert nuevo_estado.cabeza_girada_inicio is None
    assert nuevo_estado.mirada_desviada_inicio is None


def test_giro_breve_no_dispara_distraccion_cabeza():
    estado = estado_inicial_distraccion()
    for t in [0.0, 1.0]:
        estado, eventos = procesar_pose_y_mirada(
            estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
            timestamp=t, config=CONFIG,
        )
        assert eventos == []


def test_giro_exactamente_en_el_limite_no_dispara():
    estado = estado_inicial_distraccion()
    for t in [0.0, 2.0]:
        estado, eventos = procesar_pose_y_mirada(
            estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
            timestamp=t, config=CONFIG,
        )
    assert eventos == []


def test_giro_sostenido_dispara_distraccion_cabeza():
    estado = estado_inicial_distraccion()
    for t in [0.0, 1.0, 2.0]:
        estado, _ = procesar_pose_y_mirada(
            estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
            timestamp=t, config=CONFIG,
        )
    estado, eventos = procesar_pose_y_mirada(
        estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
        timestamp=2.1, config=CONFIG,
    )
    assert eventos == [EventoDistraccion(tipo="distraccion_cabeza", valor=2.1)]


def test_volver_a_mirar_al_frente_reinicia_temporizador_cabeza():
    estado = estado_inicial_distraccion()
    estado, _ = procesar_pose_y_mirada(
        estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
        timestamp=0.0, config=CONFIG,
    )
    estado, _ = procesar_pose_y_mirada(
        estado, YAW_FRENTE, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
        timestamp=0.5, config=CONFIG,
    )
    assert estado.cabeza_girada_inicio is None
    estado, _ = procesar_pose_y_mirada(
        estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
        timestamp=0.6, config=CONFIG,
    )
    assert estado.cabeza_girada_inicio == 0.6


def test_distraccion_cabeza_no_re_dispara_dentro_del_cooldown():
    estado = estado_inicial_distraccion()
    for t in [0.0, 2.1]:
        estado, eventos = procesar_pose_y_mirada(
            estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
            timestamp=t, config=CONFIG,
        )
    assert eventos == [EventoDistraccion(tipo="distraccion_cabeza", valor=2.1)]
    estado, eventos = procesar_pose_y_mirada(
        estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
        timestamp=20.0, config=CONFIG,
    )
    assert eventos == []
    estado, eventos = procesar_pose_y_mirada(
        estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
        timestamp=31.5, config=CONFIG,
    )
    assert eventos == []


def test_distraccion_cabeza_re_dispara_tras_cooldown_si_sigue_girada():
    estado = estado_inicial_distraccion()
    for t in [0.0, 2.1, 20.0, 31.5]:
        estado, eventos = procesar_pose_y_mirada(
            estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
            timestamp=t, config=CONFIG,
        )
    # El ultimo disparo de cabeza fue en t=2.1 (cabeza_girada_inicio=0.0);
    # su cooldown de 30s expira en t=32.1, asi que el llamado final debe
    # ser posterior a ese punto para volver a disparar.
    estado, eventos = procesar_pose_y_mirada(
        estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
        timestamp=32.2, config=CONFIG,
    )
    assert eventos == [EventoDistraccion(tipo="distraccion_cabeza", valor=32.2)]


def test_mirada_breve_no_dispara_distraccion_mirada():
    estado = estado_inicial_distraccion()
    for t in [0.0, 1.0]:
        estado, eventos = procesar_pose_y_mirada(
            estado, YAW_FRENTE, PITCH_FRENTE, GAZE_DESVIADO, GAZE_CENTRADO,
            timestamp=t, config=CONFIG,
        )
        assert eventos == []


def test_mirada_sostenida_dispara_distraccion_mirada():
    estado = estado_inicial_distraccion()
    for t in [0.0, 1.0, 2.0]:
        estado, _ = procesar_pose_y_mirada(
            estado, YAW_FRENTE, PITCH_FRENTE, GAZE_DESVIADO, GAZE_CENTRADO,
            timestamp=t, config=CONFIG,
        )
    estado, eventos = procesar_pose_y_mirada(
        estado, YAW_FRENTE, PITCH_FRENTE, GAZE_DESVIADO, GAZE_CENTRADO,
        timestamp=2.1, config=CONFIG,
    )
    assert eventos == [EventoDistraccion(tipo="distraccion_mirada", valor=2.1)]


def test_distraccion_mirada_no_re_dispara_dentro_del_cooldown():
    estado = estado_inicial_distraccion()
    for t in [0.0, 2.1]:
        estado, eventos = procesar_pose_y_mirada(
            estado, YAW_FRENTE, PITCH_FRENTE, GAZE_DESVIADO, GAZE_CENTRADO,
            timestamp=t, config=CONFIG,
        )
    assert eventos == [EventoDistraccion(tipo="distraccion_mirada", valor=2.1)]
    estado, eventos = procesar_pose_y_mirada(
        estado, YAW_FRENTE, PITCH_FRENTE, GAZE_DESVIADO, GAZE_CENTRADO,
        timestamp=20.0, config=CONFIG,
    )
    assert eventos == []


def test_distraccion_mirada_re_dispara_tras_cooldown_si_sigue_desviada():
    estado = estado_inicial_distraccion()
    for t in [0.0, 2.1, 20.0, 31.5]:
        estado, eventos = procesar_pose_y_mirada(
            estado, YAW_FRENTE, PITCH_FRENTE, GAZE_DESVIADO, GAZE_CENTRADO,
            timestamp=t, config=CONFIG,
        )
    estado, eventos = procesar_pose_y_mirada(
        estado, YAW_FRENTE, PITCH_FRENTE, GAZE_DESVIADO, GAZE_CENTRADO,
        timestamp=32.2, config=CONFIG,
    )
    assert eventos == [EventoDistraccion(tipo="distraccion_mirada", valor=32.2)]


def test_cabeza_y_mirada_son_independientes_entre_si():
    estado = estado_inicial_distraccion()
    for t in [0.0, 1.0, 2.0]:
        estado, _ = procesar_pose_y_mirada(
            estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
            timestamp=t, config=CONFIG,
        )
    estado, eventos = procesar_pose_y_mirada(
        estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
        timestamp=2.1, config=CONFIG,
    )
    assert eventos == [EventoDistraccion(tipo="distraccion_cabeza", valor=2.1)]
    assert estado.mirada_desviada_inicio is None
    assert estado.ultimo_disparo_mirada is None


def test_cabeza_y_mirada_pueden_dispararse_en_el_mismo_llamado():
    estado = estado_inicial_distraccion()
    for t in [0.0, 1.0, 2.0]:
        estado, _ = procesar_pose_y_mirada(
            estado, YAW_GIRADA, PITCH_FRENTE, GAZE_DESVIADO, GAZE_CENTRADO,
            timestamp=t, config=CONFIG,
        )
    estado, eventos = procesar_pose_y_mirada(
        estado, YAW_GIRADA, PITCH_FRENTE, GAZE_DESVIADO, GAZE_CENTRADO,
        timestamp=2.1, config=CONFIG,
    )
    tipos = {e.tipo for e in eventos}
    assert tipos == {"distraccion_cabeza", "distraccion_mirada"}


def test_solo_pitch_alto_tambien_activa_cabeza_girada():
    estado = estado_inicial_distraccion()
    estado, _ = procesar_pose_y_mirada(
        estado, YAW_FRENTE, 20.0, GAZE_CENTRADO, GAZE_CENTRADO,
        timestamp=0.0, config=CONFIG,
    )
    assert estado.cabeza_girada_inicio == 0.0


def test_solo_componente_vertical_de_mirada_tambien_activa_mirada_desviada():
    estado = estado_inicial_distraccion()
    estado, _ = procesar_pose_y_mirada(
        estado, YAW_FRENTE, PITCH_FRENTE, GAZE_CENTRADO, 0.75,
        timestamp=0.0, config=CONFIG,
    )
    assert estado.mirada_desviada_inicio == 0.0
```

- [ ] **Step 2: Ejecutar los tests y confirmar que fallan**

Run: `pytest tests/test_distraction_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dsd.distraction_state'`

- [ ] **Step 3: Implementar `dsd/distraction_state.py`**

```python
from dataclasses import dataclass
from typing import Optional

from dsd.config import ConfigDistraccion


@dataclass
class EstadoDistraccion:
    cabeza_girada_inicio: Optional[float] = None
    mirada_desviada_inicio: Optional[float] = None
    ultimo_disparo_cabeza: Optional[float] = None
    ultimo_disparo_mirada: Optional[float] = None


@dataclass
class EventoDistraccion:
    tipo: str
    valor: float


def estado_inicial_distraccion() -> EstadoDistraccion:
    return EstadoDistraccion()


def _procesar_temporizador_sostenido(
    condicion_activa: bool,
    inicio_actual: Optional[float],
    ultimo_disparo: Optional[float],
    timestamp: float,
    umbral_segundos: float,
    cooldown_segundos: float,
    tipo_evento: str,
) -> tuple[Optional[float], Optional[float], Optional[EventoDistraccion]]:
    if condicion_activa:
        inicio = inicio_actual if inicio_actual is not None else timestamp
    else:
        inicio = None

    duracion = (timestamp - inicio) if inicio is not None else 0.0
    evento = None
    if inicio is not None and duracion > umbral_segundos:
        en_cooldown = (
            ultimo_disparo is not None and (timestamp - ultimo_disparo) < cooldown_segundos
        )
        if not en_cooldown:
            evento = EventoDistraccion(tipo=tipo_evento, valor=duracion)
            ultimo_disparo = timestamp

    return inicio, ultimo_disparo, evento


def procesar_pose_y_mirada(
    estado: EstadoDistraccion,
    yaw: float,
    pitch: float,
    gaze_horizontal: float,
    gaze_vertical: float,
    timestamp: float,
    config: ConfigDistraccion,
) -> tuple[EstadoDistraccion, list[EventoDistraccion]]:
    eventos: list[EventoDistraccion] = []

    cabeza_girada = abs(yaw) > config.yaw_umbral_grados or abs(pitch) > config.pitch_umbral_grados
    mirada_desviada = (
        abs(gaze_horizontal - 0.5) > config.gaze_ratio_umbral
        or abs(gaze_vertical - 0.5) > config.gaze_ratio_umbral
    )

    cabeza_girada_inicio, ultimo_disparo_cabeza, evento_cabeza = _procesar_temporizador_sostenido(
        cabeza_girada,
        estado.cabeza_girada_inicio,
        estado.ultimo_disparo_cabeza,
        timestamp,
        config.distraccion_segundos,
        config.cooldown_segundos,
        "distraccion_cabeza",
    )
    if evento_cabeza is not None:
        eventos.append(evento_cabeza)

    mirada_desviada_inicio, ultimo_disparo_mirada, evento_mirada = _procesar_temporizador_sostenido(
        mirada_desviada,
        estado.mirada_desviada_inicio,
        estado.ultimo_disparo_mirada,
        timestamp,
        config.distraccion_segundos,
        config.cooldown_segundos,
        "distraccion_mirada",
    )
    if evento_mirada is not None:
        eventos.append(evento_mirada)

    nuevo_estado = EstadoDistraccion(
        cabeza_girada_inicio=cabeza_girada_inicio,
        mirada_desviada_inicio=mirada_desviada_inicio,
        ultimo_disparo_cabeza=ultimo_disparo_cabeza,
        ultimo_disparo_mirada=ultimo_disparo_mirada,
    )
    return nuevo_estado, eventos
```

- [ ] **Step 4: Ejecutar los tests y confirmar que pasan**

Run: `pytest tests/test_distraction_state.py -v`
Expected: PASS — 16 tests verdes.

- [ ] **Step 5: Commit**

```bash
git add dsd/distraction_state.py tests/test_distraction_state.py
git commit -m "feat: maquina de estados pura de distraccion (pose de cabeza + mirada)"
```

---

### Task 5: `dsd/face_mesh.py` (MODIFY) — `detectar_landmarks` reemplaza a `detectar_ojos`

**Files:**
- Modify: `dsd/face_mesh.py`

**Interfaces:**
- Consumes: nada del proyecto (solo `mediapipe`, `cv2`).
- Produces: `class ResultadoLandmarks` (dataclass) y `detectar_landmarks(frame) -> Optional[ResultadoLandmarks]` — usado por `dsd/main.py` en la Tarea 6. **`detectar_ojos` se elimina** (su único consumidor, `dsd/main.py`, se actualiza en la Tarea 6).

- [ ] **Step 1: Reemplazar `dsd/face_mesh.py` completo**

`dsd/face_mesh.py`:
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

    matriz_4x4 = resultado.facial_transformation_matrixes[0]
    matriz_rotacion = [[float(matriz_4x4[i][j]) for j in range(3)] for i in range(3)]

    return ResultadoLandmarks(
        puntos_ojo_derecho=puntos_ojo_derecho,
        puntos_ojo_izquierdo=puntos_ojo_izquierdo,
        iris_derecho=iris_derecho,
        iris_izquierdo=iris_izquierdo,
        matriz_rotacion=matriz_rotacion,
    )
```

- [ ] **Step 2: Verificación manual con las fotos ya enroladas**

Run:
```bash
source .venv/bin/activate
python3 -c "
import cv2
from dsd.face_mesh import detectar_landmarks
from dsd.head_pose import calcular_yaw_pitch

for i in range(1, 6):
    ruta = f'known_drivers/Eliecer Osorio Verdugo/foto_{i}.jpg'
    frame = cv2.imread(ruta)
    resultado = detectar_landmarks(frame)
    if resultado is None:
        print(f'foto_{i}.jpg: sin rostro detectado')
        continue
    yaw, pitch = calcular_yaw_pitch(resultado.matriz_rotacion)
    print(
        f'foto_{i}.jpg: yaw={yaw:.1f} pitch={pitch:.1f} '
        f'iris_derecho={resultado.iris_derecho} iris_izquierdo={resultado.iris_izquierdo}'
    )
"
```
Expected: para las 5 fotos, `resultado` no es `None`. `foto_2.jpg` (foto de perfil, ya identificada como tal en el sub-proyecto anterior) debiera mostrar `|yaw|` notablemente mayor que las demás fotos frontales (que deberían estar razonablemente cerca de 0). Las coordenadas de `iris_derecho`/`iris_izquierdo` deben caer dentro o muy cerca del rango de coordenadas de `puntos_ojo_derecho`/`puntos_ojo_izquierdo` respectivamente (no en otra parte del rostro) — compara visualmente los valores impresos.

**Contingencia (solo si algo de lo anterior falla):**
- Si `resultado.facial_transformation_matrixes` está vacío o lanza `AttributeError` pese a `output_facial_transformation_matrixes=True`: revisar la versión instalada de `mediapipe` (`pip show mediapipe`) y el nombre exacto del atributo en `FaceLandmarkerResult` (puede variar entre versiones); ajustar el nombre de atributo usado y volver a correr este mismo script de verificación.
- Si la extracción `matriz_4x4[i][j]` falla (p.ej. porque `matriz_4x4` es un `numpy.ndarray` con otra convención de indexado) usar `matriz_4x4[:3, :3].tolist()` en su lugar.
- Si las coordenadas de iris impresas caen claramente fuera de la región del ojo (esquina/parpado): los índices 468/473 no corresponden al centro del iris en esta versión; reemplazar por el promedio de los 4 puntos del contorno del iris (`469,470,471,472` para el derecho; `474,475,476,477` para el izquierdo) en vez de un único índice de centro.
- Si `foto_2.jpg` NO muestra un `|yaw|` claramente mayor que las demás: revisar el orden de filas/columnas usado en `calcular_yaw_pitch` (podría requerir transponer `matriz_rotacion` antes de pasarla) — la fórmula en sí ya está verificada matemáticamente en la Tarea 1 con matrices sintéticas, así que un desajuste aquí indica una convención distinta en la matriz real de Mediapipe, no un error de fórmula.

- [ ] **Step 3: Verificación manual con frame en blanco**

Run:
```bash
python3 -c "
import numpy as np
from dsd.face_mesh import detectar_landmarks
frame_vacio = np.zeros((480, 640, 3), dtype='uint8')
print(detectar_landmarks(frame_vacio))
"
```
Expected: `None`.

- [ ] **Step 4: Commit**

```bash
git add dsd/face_mesh.py
git commit -m "feat: detectar_landmarks (ojos + iris + matriz de rotacion) en una sola inferencia"
```

---

### Task 6: `dsd/main.py` (MODIFY) — integración completa

**Files:**
- Modify: `dsd/main.py`

**Interfaces:**
- Consumes: `cargar_config_distraccion` (Task 3), `estado_inicial_distraccion`/`procesar_pose_y_mirada` (Task 4), `detectar_landmarks`/`ResultadoLandmarks` (Task 5), `calcular_yaw_pitch` (Task 1), `calcular_gaze_ratio` (Task 2), y todo lo ya consumido antes (somnolencia, reconocimiento, sesión).
- Produces: comando `python -m dsd.main` — aplicación completa con reconocimiento de conductor, sesión, somnolencia y ahora también detección de distracción.

- [ ] **Step 1: Reemplazar `dsd/main.py` completo**

`dsd/main.py`:
```python
import threading
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

import cv2

from dsd.config import cargar_config, cargar_config_distraccion
from dsd.db import (
    abrir_sesion,
    cerrar_sesion,
    init_db,
    obtener_conductor_por_nombre,
    registrar_evento,
)
from dsd.distraction_state import estado_inicial_distraccion, procesar_pose_y_mirada
from dsd.drowsiness_state import estado_inicial_somnolencia, procesar_ear
from dsd.eye_metrics import calcular_ear
from dsd.face_mesh import detectar_landmarks
from dsd.gaze_metrics import calcular_gaze_ratio
from dsd.head_pose import calcular_yaw_pitch
from dsd.recognition import reconocer_conductor
from dsd.session_state import Estado, estado_inicial, procesar_deteccion

RUTA_DB = "data/app.db"
RUTA_CONFIG_SOMNOLENCIA = "config/somnolencia.yaml"
RUTA_CONFIG_DISTRACCION = "config/distraccion.yaml"

frame_actual = None
resultado_cacheado: Optional[Tuple[str, float]] = None
lock = threading.Lock()
detener = threading.Event()


def hilo_reconocimiento() -> None:
    global resultado_cacheado
    while not detener.is_set():
        with lock:
            frame = frame_actual.copy() if frame_actual is not None else None
        if frame is None:
            continue
        resultado = reconocer_conductor(frame)
        with lock:
            resultado_cacheado = resultado


def main() -> None:
    global frame_actual

    conn = init_db(RUTA_DB)
    config_somnolencia = cargar_config(RUTA_CONFIG_SOMNOLENCIA)
    config_distraccion = cargar_config_distraccion(RUTA_CONFIG_DISTRACCION)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la camara.")
        return

    hilo = threading.Thread(target=hilo_reconocimiento, daemon=True)
    hilo.start()

    estado = estado_inicial()
    estado_somnolencia = estado_inicial_somnolencia()
    estado_distraccion = estado_inicial_distraccion()
    session_id_activo = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            with lock:
                frame_actual = frame.copy()
                resultado = resultado_cacheado

            nombre_detectado = resultado[0] if resultado else None
            timestamp = time.monotonic()
            estado, eventos = procesar_deteccion(estado, nombre_detectado, timestamp)

            for evento in eventos:
                ahora_iso = datetime.now(timezone.utc).isoformat()
                if evento.tipo == "sesion_iniciada":
                    driver_id = obtener_conductor_por_nombre(conn, evento.conductor)
                    if driver_id is None:
                        print(f"Advertencia: '{evento.conductor}' no tiene registro en la base de datos, sesion no persistida.")
                        session_id_activo = None
                    else:
                        session_id_activo = abrir_sesion(conn, driver_id, ahora_iso)
                    # Reinicia el rastreo de somnolencia y distraccion: cada
                    # sesion (mismo conductor u otro) empieza con los
                    # temporizadores en blanco.
                    estado_somnolencia = estado_inicial_somnolencia()
                    estado_distraccion = estado_inicial_distraccion()
                    print(f"Sesion iniciada: {evento.conductor}")
                elif evento.tipo == "sesion_cerrada":
                    if session_id_activo is None:
                        print(f"Advertencia: sesion de '{evento.conductor}' no persistida, nada que cerrar en la base de datos.")
                    else:
                        cerrar_sesion(conn, session_id_activo, ahora_iso)
                        print(f"Sesion cerrada: {evento.conductor}")
                    session_id_activo = None

            if estado.estado == Estado.ACTIVA:
                landmarks = detectar_landmarks(frame)
                if landmarks is not None:
                    for x, y in landmarks.puntos_ojo_derecho + landmarks.puntos_ojo_izquierdo:
                        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)

                    ear_derecho = calcular_ear(landmarks.puntos_ojo_derecho)
                    ear_izquierdo = calcular_ear(landmarks.puntos_ojo_izquierdo)
                    ear_promedio = (ear_derecho + ear_izquierdo) / 2
                    estado_somnolencia, eventos_somnolencia = procesar_ear(
                        estado_somnolencia, ear_promedio, timestamp, config_somnolencia
                    )
                    for evento_somnolencia in eventos_somnolencia:
                        ahora_iso = datetime.now(timezone.utc).isoformat()
                        print(
                            f"Evento de somnolencia: {evento_somnolencia.tipo} "
                            f"(valor={evento_somnolencia.valor:.3f})"
                        )
                        if session_id_activo is not None:
                            registrar_evento(
                                conn,
                                session_id_activo,
                                evento_somnolencia.tipo,
                                evento_somnolencia.valor,
                                ahora_iso,
                            )
                        else:
                            print("Advertencia: evento de somnolencia no persistido, no hay sesion activa en la base de datos.")

                    yaw, pitch = calcular_yaw_pitch(landmarks.matriz_rotacion)
                    gaze_h_derecho, gaze_v_derecho = calcular_gaze_ratio(
                        landmarks.iris_derecho, landmarks.puntos_ojo_derecho
                    )
                    gaze_h_izquierdo, gaze_v_izquierdo = calcular_gaze_ratio(
                        landmarks.iris_izquierdo, landmarks.puntos_ojo_izquierdo
                    )
                    gaze_horizontal = (gaze_h_derecho + gaze_h_izquierdo) / 2
                    gaze_vertical = (gaze_v_derecho + gaze_v_izquierdo) / 2
                    estado_distraccion, eventos_distraccion = procesar_pose_y_mirada(
                        estado_distraccion,
                        yaw,
                        pitch,
                        gaze_horizontal,
                        gaze_vertical,
                        timestamp,
                        config_distraccion,
                    )
                    for evento_distraccion in eventos_distraccion:
                        ahora_iso = datetime.now(timezone.utc).isoformat()
                        print(
                            f"Evento de distraccion: {evento_distraccion.tipo} "
                            f"(valor={evento_distraccion.valor:.3f})"
                        )
                        if session_id_activo is not None:
                            registrar_evento(
                                conn,
                                session_id_activo,
                                evento_distraccion.tipo,
                                evento_distraccion.valor,
                                ahora_iso,
                            )
                        else:
                            print("Advertencia: evento de distraccion no persistido, no hay sesion activa en la base de datos.")

            if estado.estado == Estado.ACTIVA:
                texto = f"Sesion activa: {estado.conductor_actual}"
            else:
                texto = "Buscando conductor..."

            cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Deteccion de conductor", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detener.set()
        hilo.join(timeout=2)
        if session_id_activo is not None:
            cerrar_sesion(conn, session_id_activo, datetime.now(timezone.utc).isoformat())
            print("Sesion activa cerrada al salir.")
        cap.release()
        cv2.destroyAllWindows()
        conn.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verificación manual — escenarios completos**

Run: `source .venv/bin/activate && python -m dsd.main`

Con "Eliecer Osorio Verdugo" (ya enrolado) frente a la cámara, verificar en orden:
1. Mostrar el rostro → `Sesion iniciada: Eliecer Osorio Verdugo`.
2. Mirar al frente con normalidad por ~5 segundos → **no** debe aparecer ningún `Evento de distraccion`.
3. Girar la cabeza hacia un costado (radio/pasajero) y mantenerla girada más de 2 segundos → consola imprime `Evento de distraccion: distraccion_cabeza (valor=X.XXX)` con `X.XXX > 2.0`.
4. Volver a mirar al frente y repetir el giro varias veces dentro de los 30 segundos siguientes → **no** debe repetirse el mismo evento (cooldown activo).
5. Con la cabeza al frente, mover solo los ojos hacia un costado (sin girar la cabeza) y mantenerlos así más de 2 segundos → consola imprime `Evento de distraccion: distraccion_mirada (valor=X.XXX)`.
6. Presionar `q` con una sesión activa → `Sesion activa cerrada al salir.`

- [ ] **Step 3: Verificar persistencia de eventos en la base de datos**

Run:
```bash
sqlite3 data/app.db "SELECT id, session_id, tipo, valor, timestamp, synced FROM events ORDER BY id DESC LIMIT 10;"
```
Expected: los eventos de distracción del Step 2 aparecen con `tipo` en `{"distraccion_cabeza", "distraccion_mirada"}`, `synced = 0`, y `session_id` apuntando a la sesión activa correspondiente.

- [ ] **Step 4: Commit**

```bash
git add dsd/main.py
git commit -m "feat: integrar deteccion de distraccion (pose de cabeza + mirada) en main.py"
```

---

## Self-Review

- **Cobertura del spec:** los 6 módulos del diseño están cubiertos: `dsd/head_pose.py` (Task 1, extracción pura de yaw/pitch verificada matemáticamente con matrices sintéticas), `dsd/gaze_metrics.py` (Task 2, ratio de mirada puro con 7 tests), `dsd/config.py` extendido + `config/distraccion.yaml` (Task 3, umbrales documentados con citas a Klauer et al. 2006 y práctica DMS/Euro NCAP), `dsd/distraction_state.py` (Task 4, máquina de estados pura con temporizadores independientes para cabeza y mirada, 16 tests, sin necesidad de ventana acumulada tipo PERCLOS — justificado explícitamente en Global Constraints), `dsd/face_mesh.py` (Task 5, una sola inferencia por frame vía `detectar_landmarks`, verificado con las 5 fotos reales), `dsd/main.py` (Task 6, integración completa, verificación manual con cámara real). Ambos tipos de evento (`distraccion_cabeza`, `distraccion_mirada`) pueden dispararse en el mismo llamado, confirmado por `test_cabeza_y_mirada_pueden_dispararse_en_el_mismo_llamado`. La tabla `events` se reutiliza sin cambios de esquema, tal como especifica el diseño.
- **Placeholders:** ninguno. Todo el código de las 6 tareas está completo y fue validado numéricamente a mano por el autor del plan (yaw/pitch: verificado por trazas manuales de la descomposición de Euler para yaw=30/pitch=0, yaw=0/pitch=20, yaw=15/pitch=10 con y sin roll; gaze ratio: verificado con puntos sintéticos idénticos a los ya usados en `eye_metrics.py`; distraction_state: verificado el mismo tipo de traza de cooldown que encontró y corrigió el bug de somnolencia, esta vez sin colisión porque cabeza y mirada nunca comparten campos de estado). La única excepción deliberada es la contingencia de Task 5 sobre la convención exacta de la matriz de Mediapipe (nombre de atributo, indexado, orientación de ejes) y sobre los índices de iris (468/473), que son inherentemente inciertos sin ejecutar el código real contra la cámara — cada una incluye un paso de verificación explícito con comando exacto, salida esperada, y una ruta de corrección concreta si falla.
- **Consistencia de tipos/nombres:** `ConfigDistraccion` se usa igual en Task 3, Task 4 y Task 6. `EstadoDistraccion`/`EventoDistraccion`/`estado_inicial_distraccion`/`procesar_pose_y_mirada` se usan igual en Task 4 y Task 6. `calcular_yaw_pitch` se usa igual en Task 1, la verificación manual de Task 5 y Task 6. `calcular_gaze_ratio` se usa igual en Task 2 y Task 6. `ResultadoLandmarks`/`detectar_landmarks` se usan igual en Task 5 y Task 6. `detectar_ojos` se elimina en Task 5 y ningún task posterior lo referencia. El orden de los 6 puntos de ojo (`esquina_externa, parpado_superior_1, parpado_superior_2, esquina_interna, parpado_inferior_2, parpado_inferior_1`) es idéntico entre `eye_metrics.py` (ya existente), `gaze_metrics.py` (Task 2) y `face_mesh.py` (Task 5). Verificado sin discrepancias.

## Verificación end-to-end de esta etapa

1. Los 4 módulos con TDD (Tasks 1-4) deben pasar sus tests automatizados (`pytest tests/ -v`), sumando a los 45 tests ya existentes de los dos sub-proyectos anteriores.
2. Los módulos impuros (Tasks 5-6) requieren verificación manual con cámara real y fotos ya enroladas, siguiendo el mismo patrón ya establecido para `face_mesh.py`/`main.py` en el sub-proyecto de somnolencia.
3. Verificación final: correr `python -m dsd.main`, provocar un giro de cabeza y una desviación de mirada reales, y confirmar con `sqlite3 data/app.db` que ambos eventos quedaron persistidos correctamente en la tabla `events`.

## Actualización post-revisión final: hueco de detección + mirada durante ojos cerrados

La revisión final de todo el branch (7 commits de Tasks 1-6 + main.py) encontró 2 hallazgos Importantes:

**1. Temporizadores congelados durante huecos de detección facial.** `cabeza_girada_inicio`/`mirada_desviada_inicio` (y también `cierre_inicio` de microsueño en `drowsiness_state.py`, ya en producción desde el sub-proyecto anterior) no verificaban cuánto tiempo pasó desde el último frame procesado. Si el rostro no se detecta durante un tramo largo (frames descartados por completo, según el Global Constraint de "sin rostro se descarta") y luego se retoma con la condición ya activa, el código asumía continuidad durante todo el hueco, disparando un evento inmediato con duración inflada — el mismo tipo de falso positivo que `perclos_cobertura_minima` ya resuelve para la ventana deslizante, pero en el temporizador de sostenimiento.

**Fix:** se extrajo `dsd/sustained_timer.py` (módulo puro nuevo, con tests propios) implementando el patrón "duración continua + cooldown" de forma reutilizable, con un parámetro `hubo_hueco` explícito. Se agregó `gap_maximo_segundos: 1.0` a ambas configs (`somnolencia.yaml`, `distraccion.yaml`), documentado como salvaguarda de ingeniería. `drowsiness_state.py` (microsueño) y `distraction_state.py` (cabeza y mirada) ahora calculan `hubo_hueco` comparando el timestamp actual contra un nuevo campo `ultimo_procesado` en su estado, y lo pasan al helper compartido.

**Restricción de diseño encontrada durante la corrección:** varios tests existentes (`test_microsueno_no_re_dispara_dentro_del_cooldown`, `test_distraccion_cabeza_no_re_dispara_dentro_del_cooldown`, etc.) usaban saltos grandes de timestamp (p.ej. `0.0 → 20.0 → 31.5`) como atajo para simular "cerrado/girado de forma continua" sin llamar a la función en cada instante intermedio. Con el nuevo guard, esos saltos se interpretan (correctamente) como huecos reales, rompiendo la premisa de esos tests. Se reescribieron con muestreo continuo (`while` + paso de 0.5s, bajo `gap_maximo_segundos`), preservando la intención original de cada test. El test combinado `test_microsueno_y_perclos_pueden_dispararse_en_el_mismo_llamado` no podía reconstruirse con muestreo continuo porque microsueño (fases en 2, 32, 62...) y PERCLOS (fases en 60, 90...) tienen periodos de 30s pero fases distintas y nunca coinciden solos bajo cierre continuo — se resolvió construyendo el estado previo directamente (en vez de hacerlo evolucionar con llamadas), lo cual es válido porque `EstadoSomnolencia` es un dataclass público sin invariantes ocultos.

**2. Señal de mirada evaluada con los ojos cerrados.** Durante un microsueño sostenido, los landmarks de iris no son confiables (el párpado cubre el globo ocular), y el ratio de mirada podía dispararse fuera de rango sobre datos basura, generando un `distraccion_mirada` espurio que en realidad era un microsueño.

**Fix:** `procesar_pose_y_mirada` (Task 4) gana un parámetro `ojos_abiertos: bool` que gatea la señal de mirada (`mirada_desviada = ojos_abiertos and (...)`). `dsd/main.py` (Task 6) calcula `ojos_abiertos = ear_promedio >= config_somnolencia.ear_umbral` (reutilizando el EAR ya calculado para somnolencia, sin trabajo adicional) y lo pasa al llamar a `procesar_pose_y_mirada`.

**Verificación:** 91/91 tests pasando (18 nuevos: 10 de `sustained_timer.py`, 1 de hueco en somnolencia, 2 de hueco + gating en distracción, más los tests reescritos), sin regresión en ningún test preexistente de los 3 sub-proyectos. `python3 -c "import dsd.main"` confirma que la integración sigue siendo válida.

### Critical Files for Implementation
- /Users/cursor/Dev/eosorio/dteccion_somnolencia_distraccion/dsd/head_pose.py
- /Users/cursor/Dev/eosorio/dteccion_somnolencia_distraccion/dsd/gaze_metrics.py
- /Users/cursor/Dev/eosorio/dteccion_somnolencia_distraccion/dsd/config.py
- /Users/cursor/Dev/eosorio/dteccion_somnolencia_distraccion/config/distraccion.yaml
- /Users/cursor/Dev/eosorio/dteccion_somnolencia_distraccion/dsd/distraction_state.py
- /Users/cursor/Dev/eosorio/dteccion_somnolencia_distraccion/dsd/face_mesh.py
- /Users/cursor/Dev/eosorio/dteccion_somnolencia_distraccion/dsd/main.py
- /Users/cursor/Dev/eosorio/dteccion_somnolencia_distraccion/dsd/sustained_timer.py (nuevo, post-revision)
- /Users/cursor/Dev/eosorio/dteccion_somnolencia_distraccion/dsd/drowsiness_state.py (modificado, post-revision)
