# Detector de Somnolencia (EAR + PERCLOS) — Plan de Implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## Contexto

Este es el segundo sub-proyecto del sistema de detección de somnolencia/distracción del conductor. El primero (reconocimiento de conductor + máquina de estados de sesión) ya está completo, revisado y pusheado a `https://github.com/cursorcl/deteccion_conductores` — vive en `dsd/db.py`, `dsd/session_state.py`, `dsd/enroll.py`, `dsd/recognition.py`, `dsd/main.py`, con 18 tests pasando y verificado end-to-end con cámara real.

De los 4 detectores de comportamiento (somnolencia, distracción, celular, cigarro), se acordó abordarlos uno a la vez, empezando por **somnolencia** por ser el más crítico para seguridad y el más simple de implementar (no requiere entrenar un modelo de detección de objetos, a diferencia de celular/cigarro que usarán YOLOv8 más adelante — ya se confirmó que `ultralytics 8.4.12` funciona en este Mac con soporte MPS, y que existe una clase "Mobile phone" lista para usar en `yolov8x-oiv7.pt`, útil para ese sub-proyecto futuro).

Este sub-proyecto agrega detección de somnolencia (microsueños vía EAR sostenido, y fatiga acumulada vía PERCLOS) usando Mediapipe Face Mesh, activa solo mientras hay una sesión de conductor activa (`dsd.session_state.Estado.ACTIVA`). Solo detección y persistencia — sin alertas sonoras/visuales todavía (eso queda para un sub-proyecto futuro, junto con los otros 3 detectores, ya que comparten el mismo mecanismo de alerta). Los umbrales viven en un archivo YAML externo y documentado (no hardcodeados), para que sean ajustables sin tocar código y quede clara la referencia de investigación de cada valor.

**Nota de proceso:** este plan se generó y validó numéricamente (incluyendo una revisión que encontró y corrigió una limitación real del diseño original: evaluar PERCLOS antes de completar la ventana de 60s daba falsos positivos) usando un agente de planificación. Una vez aprobado, este documento se copiará a `docs/superpowers/plans/` del proyecto (siguiendo la convención ya establecida en el sub-proyecto anterior), no permanecerá solo en `~/.claude`.

**Goal:** Detectar somnolencia del conductor (microsueños y fatiga acumulada vía PERCLOS) usando Mediapipe Face Mesh, activo solo mientras `dsd.session_state` reporta `Estado.ACTIVA`, persistiendo eventos detectados en una nueva tabla `events` de `data/app.db`. Solo detección — sin alertas (sub-proyecto futuro).

**Architecture:** Se ejecuta en el **hilo principal** (no un hilo nuevo), sobre cada frame, porque Mediapipe Face Mesh es liviano/tiempo-real (a diferencia de DeepFace, que sigue corriendo en su propio hilo de fondo sin cambios) y necesitamos resolución temporal fina para detectar un microsueño de ~1.5s. Módulos puros (`eye_metrics.py`, `config.py`, `drowsiness_state.py`) se testean con TDD; módulos impuros (`face_mesh.py`, integración en `main.py`) se verifican manualmente con las fotos ya enroladas de "Eliecer Osorio Verdugo" y con cámara real.

**Tech Stack:** Python 3.11.11, Mediapipe Face Mesh (`mediapipe`), PyYAML, `sqlite3` (stdlib), `pytest`. Se agrega `mediapipe` y `PyYAML` a las dependencias existentes; `opencv-python` se reemplaza por `opencv-contrib-python` (ver Global Constraints — `mediapipe` requiere `opencv-contrib-python`, y tener ambos paquetes instalados a la vez genera conflicto porque los dos proveen el módulo `cv2`).

## Global Constraints

- **Archivo de configuración:** `config/somnolencia.yaml`, cargado con `dsd.config.cargar_config(path: str) -> ConfigSomnolencia`. Valores exactos:
  - `ear_umbral: 0.21`
  - `microsueno_segundos: 1.5`
  - `perclos_ventana_segundos: 60`
  - `perclos_umbral: 0.15`
  - `cooldown_segundos: 30`
- **Índices de landmarks de Mediapipe Face Mesh** (constantes en `dsd/face_mesh.py`, orden estándar de 6 puntos EAR: esquina externa, párpado superior 1, párpado superior 2, esquina interna, párpado inferior 2, párpado inferior 1):
  - Ojo derecho: `[33, 160, 158, 133, 153, 144]`
  - Ojo izquierdo: `[362, 385, 387, 263, 373, 380]`
- **Esquema exacto de la tabla `events`** (agregada a `init_db` en `dsd/db.py`, mismo archivo `data/app.db`):
  ```sql
  CREATE TABLE IF NOT EXISTS events (
      id INTEGER PRIMARY KEY,
      session_id INTEGER NOT NULL REFERENCES sessions(id),
      tipo TEXT NOT NULL,
      valor REAL NOT NULL,
      timestamp TEXT NOT NULL,
      synced INTEGER NOT NULL DEFAULT 0
  )
  ```
  - `tipo`: `"microsueno"` o `"perclos"`.
  - `valor`: para `"microsueno"`, la duración del cierre sostenido en segundos al momento del disparo; para `"perclos"`, la fracción de tiempo con ojos cerrados (0.0–1.0) al momento del disparo.
  - `synced`: `0` = no sincronizado (default), `1` = sincronizado — reservado para un sub-proyecto futuro de sincronización.
  - Nueva función: `registrar_evento(conn: sqlite3.Connection, session_id: int, tipo: str, valor: float, timestamp: str) -> int`.
- **Reinicio de estado de somnolencia:** `main.py` descarta y crea un `EstadoSomnolencia` nuevo (`estado_inicial_somnolencia()`) en el momento en que procesa el evento `sesion_iniciada` (no en `sesion_cerrada` — más defensivo: el estado limpio queda garantizado justo cuando empieza a usarse). `dsd/drowsiness_state.py` no importa ni conoce nada de `dsd.db` ni de sesiones/conductores.
- **Semántica de cooldown:** por tipo de evento (`microsueno` y `perclos` tienen cooldowns independientes), medido como tiempo transcurrido desde el último disparo de ese tipo, sin importar si los ojos se abrieron y volvieron a cerrar en el intertanto.
- **Frames sin rostro detectado durante `ACTIVA`:** se descartan por completo para la máquina de somnolencia (no se llama a `procesar_ear`, no hay muestra agregada a la ventana PERCLOS, no hay evento). Una detección faltante no es evidencia de "ojos abiertos" ni "ojos cerrados".
- **Dependencias:** `requirements.txt` pasa a:
  ```
  opencv-contrib-python==5.0.0.93
  deepface==0.0.100
  tensorflow==2.21.0
  numpy==2.4.6
  mediapipe==0.10.35
  PyYAML==6.0.3
  pytest
  ```
  (`opencv-python==5.0.0.93` se reemplaza por `opencv-contrib-python==5.0.0.93`.)
- **Rutas nuevas:** `config/somnolencia.yaml` (se versiona en git, no va en `.gitignore`); `models/face_landmarker.task` (asset binario de terceros, gitignored, descargado en Task 1).
- **Fuera de alcance:** alertas audio/visuales, sincronización real con la central (solo se prepara la columna `synced`), detectores de distracción/celular/cigarro, puerto a Orange Pi.

## Actualización post-Task 1: mediapipe 0.10.35 no tiene la API legacy `mp.solutions`

Al ejecutar Task 1, se confirmó la contingencia documentada: `mediapipe==0.10.35` no expone `mp.solutions` en absoluto (no solo `face_mesh` — el atributo `solutions` no existe). Se investigó la API real instalada y se verificó end-to-end (con una foto real de "Eliecer Osorio Verdugo") el reemplazo correcto: la **Tasks API** (`mediapipe.tasks.python.vision.FaceLandmarker`), que requiere descargar un archivo de modelo (`.task`) por separado. Esto reemplaza el diseño original de `dsd/face_mesh.py` en Task 6 y agrega un paso de descarga de modelo a Task 1. El resto del plan (Tasks 2-5, 7) **no cambia** — todos consumen `detectar_ojos(frame) -> Optional[Tuple[List[Tuple[float,float]], List[Tuple[float,float]]]]`, no la API de Mediapipe directamente, tal como anticipaba la nota de contingencia original.

Detalles verificados:
- Modelo: `face_landmarker.task` (float16, ~3.7MB), descargado de `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task` (URL oficial del modelo de Google/Mediapipe, HTTP 200 confirmado) a `models/face_landmarker.task`. Es un asset binario grande de un tercero (no código propio) — **no se versiona en git**, se descarga en setup (agregar `models/` a `.gitignore`).
- API verificada con una foto real:
  ```python
  from mediapipe.tasks.python import vision
  from mediapipe.tasks.python.core.base_options import BaseOptions

  base_options = BaseOptions(model_asset_path="models/face_landmarker.task")
  options = vision.FaceLandmarkerOptions(
      base_options=base_options,
      running_mode=vision.RunningMode.IMAGE,
      num_faces=1,
      min_face_detection_confidence=0.5,
      min_face_presence_confidence=0.5,
      min_tracking_confidence=0.5,
      output_face_blendshapes=False,
      output_facial_transformation_matrixes=False,
  )
  detector = vision.FaceLandmarker.create_from_options(options)

  mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)  # frame_rgb: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  resultado = detector.detect(mp_image)  # resultado.face_landmarks: List[List[NormalizedLandmark]]
  ```
  Confirmado con `known_drivers/Eliecer Osorio Verdugo/foto_1.jpg`: 1 rostro detectado, 478 landmarks (topología base 468 + 10 de iris; nuestros índices `[33, 160, 158, 133, 153, 144]` y `[362, 385, 387, 263, 373, 380]` siguen siendo válidos dentro de esa topología — Mediapipe mantuvo la misma numeración de landmarks del mesh clásico). Cada `NormalizedLandmark` tiene `.x`/`.y`/`.z` igual que la API legacy.
- `running_mode=vision.RunningMode.IMAGE` (no `VIDEO` ni `LIVE_STREAM`) porque llamamos a `detect()` de forma síncrona por frame, igual de simple para fotos estáticas (verificación manual) que para el loop de cámara en vivo — evita el requisito de timestamps `int64` estrictamente crecientes que exige el modo `VIDEO`.

---

### Task 1: Dependencias y estructura de carpetas

**Files:**
- Modify: `requirements.txt`
- Create: `config/` (directorio, contenido en Task 3)

**Interfaces:**
- Produces: entorno virtual con `mediapipe`, `PyYAML` y `opencv-contrib-python` instalados; `cv2` sigue funcionando para el código existente.

- [ ] **Step 1: Actualizar `requirements.txt`**

`requirements.txt`:
```
opencv-contrib-python==5.0.0.93
deepface==0.0.100
tensorflow==2.21.0
numpy==2.4.6
mediapipe==0.10.35
PyYAML==6.0.3
pytest
```

- [ ] **Step 2: Reinstalar dependencias**

Run:
```bash
cd /Users/cursor/Dev/eosorio/dteccion_somnolencia_distraccion
source .venv/bin/activate
pip uninstall -y opencv-python
pip install -r requirements.txt
```
Expected: instalación exitosa. `opencv-python` no debe quedar instalado junto a `opencv-contrib-python` (ambos proveen el módulo `cv2`).

- [ ] **Step 3: Verificar `cv2` y la API de Mediapipe instalada**

Run:
```bash
python3 -c "import cv2; print('cv2 OK:', cv2.__version__)"
python3 -c "import mediapipe as mp; print(dir(mp))"
```
Expected (ya confirmado en esta instalación de `mediapipe==0.10.35`): `cv2 OK: 5.0.0`; el segundo comando NO tiene `solutions` en la lista — solo `Image`, `ImageFormat`, `tasks`. Esta versión de mediapipe usa exclusivamente la **Tasks API** (`mediapipe.tasks.python.vision.FaceLandmarker`), ya verificada end-to-end (ver sección "Actualización post-Task 1" más arriba) y usada en `dsd/face_mesh.py` (Task 6). Si en tu entorno `mp.solutions` sí existe (versión distinta de mediapipe), la API Tasks usada en este plan sigue siendo válida igual — no depende de que la legacy exista o no.

- [ ] **Step 4: Crear el directorio `config/` y descargar el modelo de Mediapipe**

Run:
```bash
mkdir -p config
mkdir -p models
curl -sL -o models/face_landmarker.task "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
ls -la models/face_landmarker.task
```
Expected: el archivo `models/face_landmarker.task` existe y pesa aproximadamente 3.7MB (ya descargado y verificado funcional en esta máquina — si el archivo ya existe con ese tamaño, el `curl` es idempotente y no hace falta repetirlo).

- [ ] **Step 5: Agregar `models/` al `.gitignore`**

Editar `.gitignore` agregando una línea `models/` (el modelo es un asset binario de terceros descargado de Google, no código propio — no se versiona en git, igual que `known_drivers/` y `data/`).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore
git commit -m "chore: agregar mediapipe (Tasks API) y PyYAML, reemplazar opencv-python por opencv-contrib-python"
```
Nota: `models/face_landmarker.task` no se commitea (está en `.gitignore`); `config/` queda vacío por ahora (git no trackea directorios vacíos, se llena en Task 3).

---

### Task 2: `dsd/eye_metrics.py` — cálculo puro de EAR

**Files:**
- Create: `dsd/eye_metrics.py`
- Test: `tests/test_eye_metrics.py`

**Interfaces:**
- Consumes: nada (módulo puro, sin dependencias de otros módulos del proyecto).
- Produces: `calcular_ear(puntos_ojo: Sequence[Tuple[float, float]]) -> float`

- [ ] **Step 1: Escribir los tests que deben fallar**

`tests/test_eye_metrics.py`:
```python
import pytest

from dsd.eye_metrics import calcular_ear

# Puntos sintéticos en el orden estándar de 6 puntos EAR:
# (esquina_externa, parpado_superior_1, parpado_superior_2, esquina_interna,
#  parpado_inferior_2, parpado_inferior_1)
OJO_ABIERTO = [
    (0.0, 0.0), (0.3, -0.15), (0.7, -0.15), (1.0, 0.0), (0.7, 0.15), (0.3, 0.15)
]
OJO_CERRADO = [
    (0.0, 0.0), (0.3, -0.025), (0.7, -0.025), (1.0, 0.0), (0.7, 0.025), (0.3, 0.025)
]


def test_ojo_abierto_da_ear_alto():
    assert calcular_ear(OJO_ABIERTO) == pytest.approx(0.3)


def test_ojo_cerrado_da_ear_bajo():
    assert calcular_ear(OJO_CERRADO) == pytest.approx(0.05)


def test_ear_es_invariante_a_la_escala():
    ojo_escalado = [(x * 100.0, y * 100.0) for x, y in OJO_ABIERTO]
    assert calcular_ear(ojo_escalado) == pytest.approx(calcular_ear(OJO_ABIERTO))


def test_ojo_abierto_supera_umbral_tipico():
    assert calcular_ear(OJO_ABIERTO) > 0.21


def test_ojo_cerrado_no_supera_umbral_tipico():
    assert calcular_ear(OJO_CERRADO) < 0.21


def test_calcular_ear_lanza_error_si_no_son_6_puntos():
    with pytest.raises(ValueError):
        calcular_ear([(0.0, 0.0), (1.0, 1.0)])
```

- [ ] **Step 2: Ejecutar los tests y confirmar que fallan**

Run: `source .venv/bin/activate && pytest tests/test_eye_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dsd.eye_metrics'`

- [ ] **Step 3: Implementar `dsd/eye_metrics.py`**

```python
import math
from typing import Sequence, Tuple


def calcular_ear(puntos_ojo: Sequence[Tuple[float, float]]) -> float:
    """Calcula el Eye Aspect Ratio (EAR) segun Soukupova & Cech, 2016.

    `puntos_ojo` debe tener exactamente 6 puntos (x, y) en el orden estandar:
    (esquina_externa, parpado_superior_1, parpado_superior_2, esquina_interna,
    parpado_inferior_2, parpado_inferior_1).

    EAR = (dist(p2, p6) + dist(p3, p5)) / (2 * dist(p1, p4))

    El resultado es invariante a la escala (distancia de la camara al rostro):
    es una razon entre distancias, no una distancia absoluta.
    """
    if len(puntos_ojo) != 6:
        raise ValueError(
            "calcular_ear requiere exactamente 6 puntos (esquina_externa, "
            "parpado_superior_1, parpado_superior_2, esquina_interna, "
            f"parpado_inferior_2, parpado_inferior_1); se recibieron {len(puntos_ojo)}."
        )

    p1, p2, p3, p4, p5, p6 = puntos_ojo

    def distancia(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    vertical = distancia(p2, p6) + distancia(p3, p5)
    horizontal = distancia(p1, p4)

    if horizontal == 0.0:
        return 0.0

    return vertical / (2 * horizontal)
```

- [ ] **Step 4: Ejecutar los tests y confirmar que pasan**

Run: `pytest tests/test_eye_metrics.py -v`
Expected: PASS — 6 tests verdes.

- [ ] **Step 5: Commit**

```bash
git add dsd/eye_metrics.py tests/test_eye_metrics.py
git commit -m "feat: calculo puro de Eye Aspect Ratio (EAR)"
```

---

### Task 3: `dsd/config.py` — carga de configuración YAML

**Files:**
- Create: `dsd/config.py`
- Create: `config/somnolencia.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nada (módulo puro de I/O, sin dependencias de otros módulos del proyecto salvo `PyYAML`).
- Produces:
  - `class ConfigSomnolencia` (dataclass) con campos `ear_umbral: float`, `microsueno_segundos: float`, `perclos_ventana_segundos: float`, `perclos_umbral: float`, `cooldown_segundos: float`.
  - `cargar_config(path: str) -> ConfigSomnolencia`

- [ ] **Step 1: Escribir los tests que deben fallar**

`tests/test_config.py`:
```python
import pytest

from dsd.config import ConfigSomnolencia, cargar_config

YAML_VALIDO = """
ear_umbral: 0.21
microsueno_segundos: 1.5
perclos_ventana_segundos: 60
perclos_umbral: 0.15
cooldown_segundos: 30
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
    )


def test_cargar_config_convierte_valores_a_float(tmp_path):
    ruta = tmp_path / "somnolencia.yaml"
    ruta.write_text(YAML_VALIDO)

    config = cargar_config(str(ruta))

    assert isinstance(config.perclos_ventana_segundos, float)
    assert isinstance(config.cooldown_segundos, float)


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
```

- [ ] **Step 2: Ejecutar los tests y confirmar que fallan**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dsd.config'` (y el último test además fallaría por no existir `config/somnolencia.yaml` todavía).

- [ ] **Step 3: Implementar `dsd/config.py`**

```python
from dataclasses import dataclass

import yaml

CAMPOS_REQUERIDOS = (
    "ear_umbral",
    "microsueno_segundos",
    "perclos_ventana_segundos",
    "perclos_umbral",
    "cooldown_segundos",
)


@dataclass
class ConfigSomnolencia:
    ear_umbral: float
    microsueno_segundos: float
    perclos_ventana_segundos: float
    perclos_umbral: float
    cooldown_segundos: float


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
    )
```

- [ ] **Step 4: Crear `config/somnolencia.yaml`**

`config/somnolencia.yaml`:
```yaml
# Configuracion de deteccion de somnolencia (EAR + PERCLOS).
# Todos los valores provienen de literatura publica bien establecida sobre
# fatiga y microsuenos al conducir; ver el comentario sobre cada clave.
# Los indices de landmarks de ojos usados para el calculo de EAR (fijos, no
# configurables) estan documentados como constantes en dsd/face_mesh.py.

# Eye Aspect Ratio (EAR): razon geometrica calculada sobre 6 puntos del
# contorno de cada ojo (ver dsd/eye_metrics.py). Con el ojo abierto, EAR
# tipicamente esta en el rango ~0.25-0.35; al cerrarse, cae abruptamente
# hacia ~0.0-0.15. El umbral de referencia viene de Soukupova & Cech,
# "Real-Time Eye Blink Detection using Facial Landmarks" (2016), que usa
# un valor cercano a 0.2 para separar ojo abierto de ojo cerrado.
ear_umbral: 0.21

# Duracion minima continua (segundos) con EAR por debajo de `ear_umbral`
# para considerar que ocurrio un "microsueno". La literatura de fatiga al
# conducir (p.ej. Wierwille et al. 1994, investigacion patrocinada por
# NHTSA sobre monitoreo del estado del conductor) reporta que cierres de
# parpados de aproximadamente 1 a 2 segundos ya son indicativos de
# microsuenos con perdida de atencion vial. Se eligio 1.5s como punto medio
# conservador de ese rango.
microsueno_segundos: 1.5

# Tamano (segundos) de la ventana deslizante usada para calcular PERCLOS
# (PERcentage of eyelid CLOSure), la metrica de fatiga acumulada definida
# por Wierwille et al. 1994 (NHTSA). PERCLOS clasico se mide en ventanas de
# 1 a 3 minutos; se eligio 60s (el extremo mas reactivo de ese rango) para
# detectar fatiga sin esperar demasiado tiempo.
perclos_ventana_segundos: 60

# Fraccion de tiempo (0.0-1.0) dentro de la ventana anterior con los ojos
# cerrados para considerar "fatiga acumulada". La literatura de PERCLOS
# (Wierwille et al. 1994 y estudios posteriores de somnolencia al
# conducir) reporta que valores de PERCLOS por encima de aproximadamente
# 0.15 (15%) ya se asocian con niveles de somnolencia significativos.
perclos_umbral: 0.15

# Tiempo minimo (segundos) entre dos disparos consecutivos del mismo tipo
# de evento (microsueno o perclos), para evitar eventos repetidos mientras
# la condicion persiste. No proviene de un estudio especifico -- es un
# parametro de diseno para no saturar el registro de eventos; 30s permite
# que una condicion sostenida se vuelva a reportar periodicamente.
cooldown_segundos: 30
```

- [ ] **Step 5: Ejecutar los tests y confirmar que pasan**

Run: `pytest tests/test_config.py -v`
Expected: PASS — 4 tests verdes.

- [ ] **Step 6: Commit**

```bash
git add dsd/config.py config/somnolencia.yaml tests/test_config.py
git commit -m "feat: carga de configuracion YAML de umbrales de somnolencia"
```

---

### Task 4: `dsd/drowsiness_state.py` — máquina de estados pura de somnolencia

**Files:**
- Create: `dsd/drowsiness_state.py`
- Test: `tests/test_drowsiness_state.py`

**Interfaces:**
- Consumes: `ConfigSomnolencia` de `dsd.config` (Task 3) — solo como parámetro de tipo, sin importar `cargar_config`.
- Produces:
  - `class EstadoSomnolencia` (dataclass)
  - `class EventoSomnolencia` (dataclass) con campos `tipo: str`, `valor: float`
  - `estado_inicial_somnolencia() -> EstadoSomnolencia`
  - `procesar_ear(estado: EstadoSomnolencia, ear: float, timestamp: float, config: ConfigSomnolencia) -> tuple[EstadoSomnolencia, list[EventoSomnolencia]]`

- [ ] **Step 1: Escribir los tests que deben fallar**

`tests/test_drowsiness_state.py`:
```python
from dsd.config import ConfigSomnolencia
from dsd.drowsiness_state import (
    EventoSomnolencia,
    estado_inicial_somnolencia,
    procesar_ear,
)

CONFIG = ConfigSomnolencia(
    ear_umbral=0.21,
    microsueno_segundos=1.5,
    perclos_ventana_segundos=60.0,
    perclos_umbral=0.15,
    cooldown_segundos=30.0,
)

EAR_CERRADO = 0.10
EAR_ABIERTO = 0.30


def test_ojo_abierto_no_acumula_cierre():
    estado = estado_inicial_somnolencia()
    nuevo_estado, eventos = procesar_ear(estado, EAR_ABIERTO, timestamp=0.0, config=CONFIG)
    assert eventos == []
    assert nuevo_estado.cierre_inicio is None


def test_cierre_breve_no_dispara_microsueno():
    estado = estado_inicial_somnolencia()
    for t in [0.0, 0.5, 1.0]:
        estado, eventos = procesar_ear(estado, EAR_CERRADO, timestamp=t, config=CONFIG)
        assert eventos == []


def test_cierre_exactamente_en_el_limite_no_dispara():
    estado = estado_inicial_somnolencia()
    for t in [0.0, 1.5]:
        estado, eventos = procesar_ear(estado, EAR_CERRADO, timestamp=t, config=CONFIG)
    assert eventos == []


def test_cierre_sostenido_dispara_microsueno():
    estado = estado_inicial_somnolencia()
    for t in [0.0, 0.5, 1.0, 1.5]:
        estado, eventos = procesar_ear(estado, EAR_CERRADO, timestamp=t, config=CONFIG)
    estado, eventos = procesar_ear(estado, EAR_CERRADO, timestamp=1.6, config=CONFIG)
    assert eventos == [EventoSomnolencia(tipo="microsueno", valor=1.6)]


def test_microsueno_no_re_dispara_dentro_del_cooldown():
    estado = estado_inicial_somnolencia()
    for t in [0.0, 1.6]:
        estado, eventos = procesar_ear(estado, EAR_CERRADO, timestamp=t, config=CONFIG)
    estado, eventos = procesar_ear(estado, EAR_CERRADO, timestamp=20.0, config=CONFIG)
    assert eventos == []
    estado, eventos = procesar_ear(estado, EAR_CERRADO, timestamp=31.5, config=CONFIG)
    assert eventos == []


def test_microsueno_re_dispara_tras_cooldown_si_sigue_cerrado():
    estado = estado_inicial_somnolencia()
    for t in [0.0, 1.6, 20.0, 31.5]:
        estado, eventos = procesar_ear(estado, EAR_CERRADO, timestamp=t, config=CONFIG)
    estado, eventos = procesar_ear(estado, EAR_CERRADO, timestamp=31.7, config=CONFIG)
    assert eventos == [EventoSomnolencia(tipo="microsueno", valor=31.7)]


def test_apertura_de_ojos_reinicia_temporizador_microsueno():
    estado = estado_inicial_somnolencia()
    estado, _ = procesar_ear(estado, EAR_CERRADO, timestamp=0.0, config=CONFIG)
    estado, _ = procesar_ear(estado, EAR_ABIERTO, timestamp=0.5, config=CONFIG)
    assert estado.cierre_inicio is None
    estado, _ = procesar_ear(estado, EAR_CERRADO, timestamp=0.6, config=CONFIG)
    assert estado.cierre_inicio == 0.6


def test_perclos_no_evalua_antes_de_completar_ventana():
    estado = estado_inicial_somnolencia()
    eventos_perclos = []
    t = 0.0
    while t < 60.0:
        estado, eventos = procesar_ear(estado, EAR_CERRADO, timestamp=t, config=CONFIG)
        eventos_perclos += [e for e in eventos if e.tipo == "perclos"]
        t += 1.0
    assert eventos_perclos == []


def test_perclos_dispara_cuando_fraccion_cerrada_supera_umbral():
    estado = estado_inicial_somnolencia()
    eventos_perclos = []
    t = 0.0
    while t <= 65.0:
        estado, eventos = procesar_ear(estado, EAR_CERRADO, timestamp=t, config=CONFIG)
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
        estado, eventos = procesar_ear(estado, ear, timestamp=t, config=CONFIG)
        eventos_perclos += [e for e in eventos if e.tipo == "perclos"]
        t += 1.0
    assert eventos_perclos == []


def test_muestras_antiguas_se_recortan_fuera_de_la_ventana():
    estado = estado_inicial_somnolencia()
    t = 0.0
    while t <= 65.0:
        estado, _ = procesar_ear(estado, EAR_CERRADO, timestamp=t, config=CONFIG)
        t += 1.0
    assert all(
        m.timestamp >= t - 1.0 - CONFIG.perclos_ventana_segundos for m in estado.muestras
    )


def test_estado_inicial_no_tiene_muestras():
    estado = estado_inicial_somnolencia()
    assert estado.muestras == []
    assert estado.cierre_inicio is None


def test_microsueno_y_perclos_pueden_dispararse_en_el_mismo_llamado():
    estado = estado_inicial_somnolencia()
    t = 0.0
    while t < 60.0:
        estado, _ = procesar_ear(estado, EAR_CERRADO, timestamp=t, config=CONFIG)
        t += 1.0
    estado, eventos = procesar_ear(estado, EAR_CERRADO, timestamp=61.6, config=CONFIG)
    tipos = {e.tipo for e in eventos}
    assert tipos == {"microsueno", "perclos"}
```

- [ ] **Step 2: Ejecutar los tests y confirmar que fallan**

Run: `pytest tests/test_drowsiness_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dsd.drowsiness_state'`

- [ ] **Step 3: Implementar `dsd/drowsiness_state.py`**

```python
from dataclasses import dataclass, field
from typing import List, Optional

from dsd.config import ConfigSomnolencia


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


def procesar_ear(
    estado: EstadoSomnolencia,
    ear: float,
    timestamp: float,
    config: ConfigSomnolencia,
) -> tuple[EstadoSomnolencia, list[EventoSomnolencia]]:
    eventos: List[EventoSomnolencia] = []
    cerrado = ear < config.ear_umbral

    # --- Microsueno: temporizador de cierre continuo ---
    if cerrado:
        cierre_inicio = estado.cierre_inicio if estado.cierre_inicio is not None else timestamp
    else:
        cierre_inicio = None

    duracion_cierre = (timestamp - cierre_inicio) if cierre_inicio is not None else 0.0
    ultimo_disparo_microsueno = estado.ultimo_disparo_microsueno
    if cierre_inicio is not None and duracion_cierre > config.microsueno_segundos:
        en_cooldown = (
            ultimo_disparo_microsueno is not None
            and (timestamp - ultimo_disparo_microsueno) < config.cooldown_segundos
        )
        if not en_cooldown:
            eventos.append(EventoSomnolencia(tipo="microsueno", valor=duracion_cierre))
            ultimo_disparo_microsueno = timestamp

    # --- PERCLOS: ventana deslizante ---
    primer_timestamp = estado.primer_timestamp if estado.primer_timestamp is not None else timestamp
    muestras = [
        m for m in estado.muestras if m.timestamp >= timestamp - config.perclos_ventana_segundos
    ]
    muestras.append(Muestra(timestamp=timestamp, cerrado=cerrado))

    ultimo_disparo_perclos = estado.ultimo_disparo_perclos
    ventana_cubierta = (timestamp - primer_timestamp) >= config.perclos_ventana_segundos
    if ventana_cubierta and len(muestras) >= 2:
        perclos = _calcular_perclos(muestras)
        if perclos >= config.perclos_umbral:
            en_cooldown = (
                ultimo_disparo_perclos is not None
                and (timestamp - ultimo_disparo_perclos) < config.cooldown_segundos
            )
            if not en_cooldown:
                eventos.append(EventoSomnolencia(tipo="perclos", valor=perclos))
                ultimo_disparo_perclos = timestamp

    nuevo_estado = EstadoSomnolencia(
        muestras=muestras,
        cierre_inicio=cierre_inicio,
        ultimo_disparo_microsueno=ultimo_disparo_microsueno,
        ultimo_disparo_perclos=ultimo_disparo_perclos,
        primer_timestamp=primer_timestamp,
    )
    return nuevo_estado, eventos
```

- [ ] **Step 4: Ejecutar los tests y confirmar que pasan**

Run: `pytest tests/test_drowsiness_state.py -v`
Expected: PASS — 13 tests verdes.

- [ ] **Step 5: Commit**

```bash
git add dsd/drowsiness_state.py tests/test_drowsiness_state.py
git commit -m "feat: maquina de estados pura de somnolencia (microsueno + PERCLOS)"
```

---

### Task 5: `dsd/db.py` (MODIFY) — tabla `events` y `registrar_evento`

**Files:**
- Modify: `dsd/db.py`
- Modify: `tests/test_db.py`

**Interfaces:**
- Consumes: nada nuevo.
- Produces:
  - `init_db` ahora también crea la tabla `events`.
  - `registrar_evento(conn: sqlite3.Connection, session_id: int, tipo: str, valor: float, timestamp: str) -> int`

- [ ] **Step 1: Escribir los tests que deben fallar (reemplazar `tests/test_db.py` completo)**

`tests/test_db.py`:
```python
import sqlite3

import pytest

from dsd.db import (
    abrir_sesion,
    cerrar_sesion,
    crear_conductor,
    init_db,
    obtener_conductor_por_nombre,
    registrar_evento,
)


@pytest.fixture
def conn():
    return init_db(":memory:")


def test_init_db_crea_tablas(conn):
    tablas = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {"drivers", "sessions", "events"} <= tablas


def test_crear_conductor_retorna_id(conn):
    driver_id = crear_conductor(conn, "Juan", "2026-07-04T10:00:00")
    assert isinstance(driver_id, int)


def test_crear_conductor_nombre_duplicado_lanza_error(conn):
    crear_conductor(conn, "Juan", "2026-07-04T10:00:00")
    with pytest.raises(sqlite3.IntegrityError):
        crear_conductor(conn, "Juan", "2026-07-04T10:05:00")


def test_obtener_conductor_por_nombre_encontrado(conn):
    driver_id = crear_conductor(conn, "Juan", "2026-07-04T10:00:00")
    assert obtener_conductor_por_nombre(conn, "Juan") == driver_id


def test_obtener_conductor_por_nombre_no_encontrado(conn):
    assert obtener_conductor_por_nombre(conn, "Desconocido") is None


def test_abrir_sesion_queda_sin_end_time(conn):
    driver_id = crear_conductor(conn, "Juan", "2026-07-04T10:00:00")
    session_id = abrir_sesion(conn, driver_id, "2026-07-04T10:01:00")
    row = conn.execute(
        "SELECT end_time FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    assert row[0] is None


def test_cerrar_sesion_setea_end_time(conn):
    driver_id = crear_conductor(conn, "Juan", "2026-07-04T10:00:00")
    session_id = abrir_sesion(conn, driver_id, "2026-07-04T10:01:00")
    cerrar_sesion(conn, session_id, "2026-07-04T10:15:00")
    row = conn.execute(
        "SELECT end_time FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    assert row[0] == "2026-07-04T10:15:00"


def test_registrar_evento_retorna_id(conn):
    driver_id = crear_conductor(conn, "Juan", "2026-07-04T10:00:00")
    session_id = abrir_sesion(conn, driver_id, "2026-07-04T10:01:00")
    evento_id = registrar_evento(conn, session_id, "microsueno", 1.8, "2026-07-04T10:02:00")
    assert isinstance(evento_id, int)


def test_registrar_evento_synced_por_defecto_en_cero(conn):
    driver_id = crear_conductor(conn, "Juan", "2026-07-04T10:00:00")
    session_id = abrir_sesion(conn, driver_id, "2026-07-04T10:01:00")
    evento_id = registrar_evento(conn, session_id, "perclos", 0.22, "2026-07-04T10:02:00")
    row = conn.execute("SELECT synced FROM events WHERE id = ?", (evento_id,)).fetchone()
    assert row[0] == 0


def test_registrar_evento_guarda_los_valores_correctos(conn):
    driver_id = crear_conductor(conn, "Juan", "2026-07-04T10:00:00")
    session_id = abrir_sesion(conn, driver_id, "2026-07-04T10:01:00")
    evento_id = registrar_evento(conn, session_id, "microsueno", 1.8, "2026-07-04T10:02:00")
    row = conn.execute(
        "SELECT session_id, tipo, valor, timestamp FROM events WHERE id = ?", (evento_id,)
    ).fetchone()
    assert row == (session_id, "microsueno", 1.8, "2026-07-04T10:02:00")
```

- [ ] **Step 2: Ejecutar los tests y confirmar que fallan**

Run: `pytest tests/test_db.py -v`
Expected: FAIL — `ImportError: cannot import name 'registrar_evento' from 'dsd.db'`

- [ ] **Step 3: Modificar `dsd/db.py`**

`dsd/db.py`:
```python
import sqlite3
from typing import Optional


def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS drivers (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY,
            driver_id INTEGER NOT NULL REFERENCES drivers(id),
            start_time TEXT NOT NULL,
            end_time TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY,
            session_id INTEGER NOT NULL REFERENCES sessions(id),
            tipo TEXT NOT NULL,
            valor REAL NOT NULL,
            timestamp TEXT NOT NULL,
            synced INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.commit()
    return conn


def crear_conductor(conn: sqlite3.Connection, name: str, created_at: str) -> int:
    cursor = conn.execute(
        "INSERT INTO drivers (name, created_at) VALUES (?, ?)", (name, created_at)
    )
    conn.commit()
    return cursor.lastrowid


def obtener_conductor_por_nombre(conn: sqlite3.Connection, name: str) -> Optional[int]:
    row = conn.execute("SELECT id FROM drivers WHERE name = ?", (name,)).fetchone()
    return row[0] if row else None


def abrir_sesion(conn: sqlite3.Connection, driver_id: int, start_time: str) -> int:
    cursor = conn.execute(
        "INSERT INTO sessions (driver_id, start_time, end_time) VALUES (?, ?, NULL)",
        (driver_id, start_time),
    )
    conn.commit()
    return cursor.lastrowid


def cerrar_sesion(conn: sqlite3.Connection, session_id: int, end_time: str) -> None:
    conn.execute("UPDATE sessions SET end_time = ? WHERE id = ?", (end_time, session_id))
    conn.commit()


def registrar_evento(
    conn: sqlite3.Connection,
    session_id: int,
    tipo: str,
    valor: float,
    timestamp: str,
) -> int:
    cursor = conn.execute(
        "INSERT INTO events (session_id, tipo, valor, timestamp, synced) VALUES (?, ?, ?, ?, 0)",
        (session_id, tipo, valor, timestamp),
    )
    conn.commit()
    return cursor.lastrowid
```

- [ ] **Step 4: Ejecutar los tests y confirmar que pasan**

Run: `pytest tests/test_db.py -v`
Expected: PASS — 10 tests verdes.

- [ ] **Step 5: Commit**

```bash
git add dsd/db.py tests/test_db.py
git commit -m "feat: tabla events y registrar_evento para eventos de somnolencia"
```

---

### Task 6: `dsd/face_mesh.py` — wrapper de Mediapipe Face Landmarker (Tasks API)

**Files:**
- Create: `dsd/face_mesh.py`

**Interfaces:**
- Consumes: nada del proyecto (solo `mediapipe`, `cv2`); requiere que `models/face_landmarker.task` exista (descargado en Task 1, Step 4).
- Produces: `detectar_ojos(frame) -> Optional[Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]` — usado por `dsd/main.py` en Task 7.

**Nota:** este módulo usa la Tasks API (`mediapipe.tasks.python.vision.FaceLandmarker`), no la API legacy `mp.solutions.face_mesh` (removida en la versión de `mediapipe` instalada en este proyecto — ver "Actualización post-Task 1" al inicio del plan). El código de abajo ya fue verificado end-to-end contra una foto real.

- [ ] **Step 1: Implementar `dsd/face_mesh.py`**

```python
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
    output_facial_transformation_matrixes=False,
)
_detector = vision.FaceLandmarker.create_from_options(_options)


def detectar_ojos(frame) -> Optional[Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]:
    try:
        alto, ancho = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        resultado = _detector.detect(imagen_mp)
    except Exception:
        return None

    if not resultado.face_landmarks:
        return None

    landmarks = resultado.face_landmarks[0]

    def punto(indice: int) -> Tuple[float, float]:
        lm = landmarks[indice]
        return (lm.x * ancho, lm.y * alto)

    puntos_ojo_derecho = [punto(i) for i in INDICES_OJO_DERECHO]
    puntos_ojo_izquierdo = [punto(i) for i in INDICES_OJO_IZQUIERDO]
    return puntos_ojo_derecho, puntos_ojo_izquierdo
```

- [ ] **Step 2: Verificación manual con las fotos ya enroladas**

Run (repetir para `foto_1.jpg` hasta `foto_5.jpg` cambiando el nombre de archivo):
```bash
source .venv/bin/activate
python3 -c "
import cv2
from dsd.eye_metrics import calcular_ear
from dsd.face_mesh import detectar_ojos

frame = cv2.imread('known_drivers/Eliecer Osorio Verdugo/foto_1.jpg')
resultado = detectar_ojos(frame)
print('Resultado:', 'None' if resultado is None else 'rostro detectado')
if resultado is not None:
    puntos_od, puntos_oi = resultado
    print('EAR ojo derecho:', calcular_ear(puntos_od))
    print('EAR ojo izquierdo:', calcular_ear(puntos_oi))
"
```
Expected: para las 5 fotos, `resultado` no es `None` (Mediapipe detecta el rostro en fotos frontales de enrolamiento); los valores de EAR impresos están en un rango plausible (aprox. 0.2–0.4 para fotos con ojos abiertos — está bien si alguna foto puntual muestra un valor más bajo por un parpadeo momentáneo al capturarla, o por ser la foto de perfil `foto_2.jpg`).

- [ ] **Step 3: Verificación manual con cámara en vivo**

Run:
```bash
python3 -c "
import cv2
from dsd.face_mesh import detectar_ojos
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()
print(detectar_ojos(frame))
"
```
Con tu rostro frente a la cámara, expected: tupla de dos listas de 6 puntos cada una (no `None`). Repite tapando la cámara o sin rostro en el encuadre, expected: `None`.

- [ ] **Step 4: Commit**

```bash
git add dsd/face_mesh.py
git commit -m "feat: wrapper de Mediapipe Face Landmarker (Tasks API) para landmarks de ojos"
```

---

### Task 7: `dsd/main.py` (MODIFY) — integración completa

**Files:**
- Modify: `dsd/main.py`

**Interfaces:**
- Consumes: `cargar_config` (Task 3), `estado_inicial_somnolencia`/`procesar_ear` (Task 4), `registrar_evento` (Task 5), `detectar_ojos` (Task 6), `calcular_ear` (Task 2), y todo lo ya consumido antes (`reconocer_conductor`, `procesar_deteccion`, `init_db`/`abrir_sesion`/`cerrar_sesion`/`obtener_conductor_por_nombre`).
- Produces: comando `python -m dsd.main` — aplicación completa con reconocimiento de conductor, máquina de estados de sesión y ahora también detección de somnolencia con persistencia de eventos.

- [ ] **Step 1: Reemplazar `dsd/main.py` completo**

`dsd/main.py`:
```python
import threading
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

import cv2

from dsd.config import cargar_config
from dsd.db import (
    abrir_sesion,
    cerrar_sesion,
    init_db,
    obtener_conductor_por_nombre,
    registrar_evento,
)
from dsd.drowsiness_state import estado_inicial_somnolencia, procesar_ear
from dsd.eye_metrics import calcular_ear
from dsd.face_mesh import detectar_ojos
from dsd.recognition import reconocer_conductor
from dsd.session_state import Estado, estado_inicial, procesar_deteccion

RUTA_DB = "data/app.db"
RUTA_CONFIG_SOMNOLENCIA = "config/somnolencia.yaml"

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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la camara.")
        return

    hilo = threading.Thread(target=hilo_reconocimiento, daemon=True)
    hilo.start()

    estado = estado_inicial()
    estado_somnolencia = estado_inicial_somnolencia()
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
                    # Reinicia el rastreo de somnolencia: cada sesion (mismo
                    # conductor u otro) empieza con el temporizador de
                    # microsueno y la ventana de PERCLOS en blanco.
                    estado_somnolencia = estado_inicial_somnolencia()
                    print(f"Sesion iniciada: {evento.conductor}")
                elif evento.tipo == "sesion_cerrada":
                    if session_id_activo is None:
                        print(f"Advertencia: sesion de '{evento.conductor}' no persistida, nada que cerrar en la base de datos.")
                    else:
                        cerrar_sesion(conn, session_id_activo, ahora_iso)
                        print(f"Sesion cerrada: {evento.conductor}")
                    session_id_activo = None

            if estado.estado == Estado.ACTIVA:
                puntos_ojos = detectar_ojos(frame)
                if puntos_ojos is not None:
                    puntos_ojo_derecho, puntos_ojo_izquierdo = puntos_ojos
                    ear_derecho = calcular_ear(puntos_ojo_derecho)
                    ear_izquierdo = calcular_ear(puntos_ojo_izquierdo)
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
1. Mostrar el rostro → consola imprime `Sesion iniciada: Eliecer Osorio Verdugo`, overlay muestra `Sesion activa: Eliecer Osorio Verdugo`.
2. Mantener los ojos abiertos con normalidad por ~10 segundos → **no** debe imprimirse ningún `Evento de somnolencia`.
3. Cerrar los ojos deliberadamente y mantenerlos cerrados más de 1.5 segundos seguidos → consola imprime `Evento de somnolencia: microsueno (valor=X.XXX)` con `X.XXX > 1.5`.
4. Repetir el cierre de ojos varias veces dentro de los 30 segundos siguientes al paso 3 → **no** debe volver a imprimirse `microsueno` (cooldown activo) hasta que pasen los 30 segundos.
5. (Opcional, requiere paciencia) mantener los ojos mayormente cerrados/entrecerrados durante 60+ segundos → eventualmente debe imprimirse `Evento de somnolencia: perclos (valor=0.XXX)` con `valor >= 0.15`.
6. Salir del cuadro por más de 10 segundos → consola imprime `Sesion cerrada: Eliecer Osorio Verdugo`.
7. Volver a mostrar el rostro → nueva sesión (`Sesion iniciada` de nuevo); confirmar que el temporizador de somnolencia arrancó en blanco (no debe dispararse `microsueno` inmediatamente solo por reabrir la sesión).
8. Presionar `q` con una sesión activa → consola imprime `Sesion activa cerrada al salir.`

- [ ] **Step 3: Verificar persistencia de eventos en la base de datos**

Run:
```bash
sqlite3 data/app.db "SELECT id, session_id, tipo, valor, timestamp, synced FROM events ORDER BY id;"
```
Expected: cada evento impreso en consola en el Step 2 aparece como una fila, con `synced = 0` en todas, `session_id` apuntando a la sesión activa correspondiente, y `tipo` en `{"microsueno", "perclos"}`.

- [ ] **Step 4: Commit**

```bash
git add dsd/main.py
git commit -m "feat: integrar deteccion de somnolencia (EAR + PERCLOS) en main.py"
```

---

## Self-Review

- **Cobertura del spec:** los 6 módulos propuestos están cubiertos: `dsd/config.py` (Task 3, YAML documentado con citas a Wierwille et al. 1994 y Soukupová & Čech 2016), `dsd/eye_metrics.py` (Task 2, EAR puro con 6 tests incluyendo invariancia de escala), `dsd/drowsiness_state.py` (Task 4, microsueño + PERCLOS combinados en un solo módulo puro con 13 tests, ventana ponderada por tiempo, cooldown por tipo de evento, corrección de "ventana no completada" agregada), `dsd/face_mesh.py` (Task 6, índices de landmarks documentados como constantes, modelo creado una sola vez a nivel de módulo, verificado con las 5 fotos reales de "Eliecer Osorio Verdugo"), `dsd/db.py` modificado (Task 5, tabla `events` exacta con `synced`, `registrar_evento`, 10 tests), `dsd/main.py` modificado (Task 7, evaluación solo en `Estado.ACTIVA`, reinicio de estado de somnolencia por sesión, persistencia + print en cada evento, verificación manual con cámara real). Ambos comportamientos (microsueño y PERCLOS) están activos simultáneamente en cada llamada a `procesar_ear`, confirmado por el test `test_microsueno_y_perclos_pueden_dispararse_en_el_mismo_llamado`. Nada del spec quedó sin tarea.
- **Placeholders:** ninguno. Todo el código de los 7 módulos está completo y fue validado numéricamente por el agente de planificación (EAR: 0.3/0.05/invariante a escala; microsueño: dispara en 1.6s no en 1.5s, respeta cooldown; PERCLOS: no dispara antes de completar la ventana de 60s, dispara con 100% cerrado, no dispara con 10% cerrado; `events`: inserción y lectura verificadas con sqlite3 real). La única excepción deliberada es la nota de contingencia en Task 1/Task 6 sobre la API legacy de Mediapipe, que es condicional por naturaleza y ya incluye un paso de verificación explícito con comando exacto y salida esperada para resolverla antes de continuar.
- **Consistencia de tipos/nombres:** `ConfigSomnolencia` se usa igual en Task 3, Task 4 y Task 7. `EstadoSomnolencia`/`EventoSomnolencia`/`estado_inicial_somnolencia`/`procesar_ear` se usan igual en Task 4 y Task 7. `calcular_ear` se usa igual en Task 2, la verificación manual de Task 6 y Task 7. `detectar_ojos` se usa igual en Task 6 y Task 7. `registrar_evento(conn, session_id, tipo, valor, timestamp)` se usa igual en Task 5 y Task 7. El esquema de `events` (columnas y orden) es idéntico entre el DDL de Global Constraints, `dsd/db.py` y los tests de Task 5. Verificado sin discrepancias.

## Nota de seguridad del proceso

Durante la validación de este plan, el agente de planificación reportó haber encontrado un intento de inyección de prompt embebido en el resultado de lectura de `dsd/session_state.py` (texto con formato de system-reminder falso, pidiendo escribir archivos con una herramienta que no tenía disponible). El agente lo ignoró correctamente y seguro con las instrucciones reales. Se verificó `dsd/session_state.py` directamente después de esto y su contenido está limpio, sin ningún texto inyectado — coincide exactamente con el código construido en el sub-proyecto anterior. No se tomó ninguna acción basada en esa instrucción falsa.

## Verificación end-to-end de esta etapa

1. Los 4 módulos con TDD (Tasks 2-5) deben pasar sus tests automatizados (`pytest tests/ -v`), sumando a los 18 tests ya existentes del sub-proyecto anterior.
2. Los módulos impuros (Tasks 6-7) requieren verificación manual con cámara real, ya que dependen de Mediapipe/hardware — no se pueden automatizar sin mockear la cámara, lo cual iría en contra del patrón ya establecido (`recognition.py`, `main.py` tampoco tienen tests automatizados).
3. Verificación final: correr `python -m dsd.main`, provocar un microsueño real cerrando los ojos, y confirmar con `sqlite3 data/app.db` que el evento quedó persistido correctamente en la tabla `events`.

### Critical Files for Implementation
- /Users/cursor/Dev/eosorio/dteccion_somnolencia_distraccion/dsd/drowsiness_state.py
- /Users/cursor/Dev/eosorio/dteccion_somnolencia_distraccion/dsd/config.py
- /Users/cursor/Dev/eosorio/dteccion_somnolencia_distraccion/config/somnolencia.yaml
- /Users/cursor/Dev/eosorio/dteccion_somnolencia_distraccion/dsd/face_mesh.py
- /Users/cursor/Dev/eosorio/dteccion_somnolencia_distraccion/dsd/main.py
- /Users/cursor/Dev/eosorio/dteccion_somnolencia_distraccion/dsd/db.py
