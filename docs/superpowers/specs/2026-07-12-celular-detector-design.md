# Detector de Uso de Celular — Diseño

## Contexto

Cuarto sub-proyecto de detección de comportamiento del conductor (después de somnolencia, distracción y bostezo), y el primero que **no** se resuelve con landmarks faciales de Mediapipe: detectar un celular en el frame es un problema de detección de objetos, no de geometría facial.

Se investigó y confirmó en esta misma sesión (antes de escribir este spec) que la instalación de `mediapipe==0.10.35` ya incluye `vision.ObjectDetector` (misma familia de API de Tasks que `FaceLandmarker`), y que su modelo estándar pre-entrenado sobre COCO (`efficientdet_lite0`) incluye `"cell phone"` como una de sus 80 clases. Se descargó el modelo desde la URL oficial de Google/Mediapipe (HTTP 200 confirmado), se cargó con `ObjectDetector` y se corrió una detección real contra un frame de cámara — la API respondió con la forma esperada (`detections[].categories[].category_name/.score`). No se requiere entrenar ni buscar un modelo propio para este sub-proyecto.

**Cigarro queda explícitamente fuera de este spec.** COCO no tiene una clase de cigarrillo, así que ese sub-proyecto va a necesitar su propia investigación de modelo (entrenar uno, o evaluar un modelo pre-entrenado de otra fuente) antes de poder especificarse con el mismo nivel de detalle. La arquitectura de este spec (un módulo de detección de objetos + un detector de "comportamiento sostenido" sobre esas detecciones) está pensada para ser reutilizable cuando ese sub-proyecto arranque, pero no se diseña cigarro aquí.

Igual que los detectores anteriores, el alcance es **solo detección + persistencia**, sin alertas. Activo solo mientras `dsd.session_state` reporta `Estado.ACTIVA`.

## Objetivo

Detectar cuándo el conductor está usando el celular: el modelo de objetos detecta la etiqueta `"cell phone"` en **cualquier parte del frame** (no se exige proximidad a la cara — más simple, aunque puede dar falsos positivos si hay un celular fijo y visible en un soporte dentro del encuadre; decisión consciente, no una limitación no evaluada) con confianza sobre un umbral, sostenido durante un tiempo mínimo. Dispara un evento `uso_celular` con el mismo patrón de temporizador sostenido + cooldown que ya usan microsueño y bostezo.

A diferencia de la detección de landmarks faciales (que corre solo cuando hay landmarks, dentro de `if landmarks is not None`), la detección de objetos es una señal independiente: corre en paralelo a la detección facial, directamente bajo el mismo gate `Estado.ACTIVA`, sin depender de que se haya detectado una cara ese frame — un conductor mirando el celular hacia abajo puede fallar la detección facial pero seguir siendo detectable como "celular en el frame".

## Arquitectura

```
models/efficientdet_lite0.tflite  (NUEVO, gitignored) — modelo COCO pre-entrenado
dsd/object_detection.py           (NUEVO, impuro) — wrapper de ObjectDetector
dsd/phone_state.py                (NUEVO, puro) — temporizador sostenido sobre "cell phone"
dsd/config.py                     (MODIFICADO) — + ConfigCelular, cargar_config_celular
config/celular.yaml               (NUEVO) — umbrales documentados
dsd/db.py                         (SIN CAMBIOS) — tabla `events` ya es genérica
dsd/main.py                       (MODIFICADO) — integra deteccion de celular junto al resto
```

### Setup: descarga del modelo

```bash
curl -sL -o models/efficientdet_lite0.tflite "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite"
```

URL oficial verificada (HTTP 200) en esta sesión. Asset binario de terceros (~13.8MB) — no se versiona en git, ya cubierto por `models/` en `.gitignore`. Mismo patrón que `models/face_landmarker.task`.

### `dsd/object_detection.py` (nuevo, impuro — wrapper de Mediapipe)

```python
from dataclasses import dataclass
from typing import List

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

RUTA_MODELO = "models/efficientdet_lite0.tflite"

# El detector se crea una sola vez al importar el modulo, igual que el
# FaceLandmarker en dsd/face_mesh.py. El umbral de confianza aplicado aqui
# (score_threshold) es solo un piso bajo a nivel del modelo para no
# acumular ruido de detecciones muy debiles -- la decision real de "esto
# cuenta como celular" vive en dsd/phone_state.py (config.confianza_umbral),
# igual que ear_umbral vive en drowsiness_state.py y no en face_mesh.py.
_base_options = BaseOptions(model_asset_path=RUTA_MODELO)
_options = vision.ObjectDetectorOptions(
    base_options=_base_options,
    running_mode=vision.RunningMode.IMAGE,
    max_results=10,
    score_threshold=0.2,
)
_detector = vision.ObjectDetector.create_from_options(_options)


@dataclass
class ObjetoDetectado:
    etiqueta: str
    confianza: float


def detectar_objetos(frame) -> List[ObjetoDetectado]:
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        resultado = _detector.detect(imagen_mp)
    except Exception:
        return []

    objetos = []
    for deteccion in resultado.detections:
        for categoria in deteccion.categories:
            objetos.append(
                ObjetoDetectado(etiqueta=categoria.category_name, confianza=categoria.score)
            )
    return objetos
```

### `dsd/phone_state.py` (nuevo, puro)

Mismo patrón que `dsd/distraction_state.py`: un único temporizador de sostenimiento (`dsd/sustained_timer.py`, reutilizado sin cambios) sobre la condición binaria "hay un `cell phone` con confianza suficiente en este frame".

```python
from dataclasses import dataclass
from typing import List, Optional

from dsd.config import ConfigCelular
from dsd.object_detection import ObjetoDetectado
from dsd.sustained_timer import EstadoTemporizadorSostenido, procesar_temporizador_sostenido

ETIQUETA_CELULAR = "cell phone"


@dataclass
class EstadoCelular:
    deteccion_inicio: Optional[float] = None
    ultimo_disparo: Optional[float] = None
    ultimo_procesado: Optional[float] = None


@dataclass
class EventoCelular:
    tipo: str
    valor: float


def estado_inicial_celular() -> EstadoCelular:
    return EstadoCelular()


def procesar_objetos(
    estado: EstadoCelular,
    objetos: List[ObjetoDetectado],
    timestamp: float,
    config: ConfigCelular,
) -> tuple[EstadoCelular, list[EventoCelular]]:
    eventos: List[EventoCelular] = []

    hubo_hueco = (
        estado.ultimo_procesado is not None
        and (timestamp - estado.ultimo_procesado) > config.gap_maximo_segundos
    )

    celular_detectado = any(
        o.etiqueta == ETIQUETA_CELULAR and o.confianza >= config.confianza_umbral
        for o in objetos
    )

    temporizador = EstadoTemporizadorSostenido(
        inicio=estado.deteccion_inicio, ultimo_disparo=estado.ultimo_disparo
    )
    temporizador, valor = procesar_temporizador_sostenido(
        temporizador,
        celular_detectado,
        hubo_hueco,
        timestamp,
        config.celular_segundos,
        config.cooldown_segundos,
    )
    if valor is not None:
        eventos.append(EventoCelular(tipo="uso_celular", valor=valor))

    nuevo_estado = EstadoCelular(
        deteccion_inicio=temporizador.inicio,
        ultimo_disparo=temporizador.ultimo_disparo,
        ultimo_procesado=timestamp,
    )
    return nuevo_estado, eventos
```

### `config/celular.yaml` (nuevo) + `dsd/config.py` (extendido)

```yaml
# Configuracion de deteccion de uso de celular.
# La etiqueta detectada ("cell phone") viene del modelo pre-entrenado sobre
# COCO (efficientdet_lite0); ver dsd/object_detection.py y dsd/phone_state.py.

# Confianza minima (0.0-1.0) de la deteccion de "cell phone" del modelo de
# objetos para considerarla real y no ruido. Decision de ingenieria --
# se calibrara con verificacion manual real, igual que mar_umbral y
# gaze_ratio_umbral en los detectores de somnolencia/distraccion.
confianza_umbral: 0.5

# Duracion minima continua (segundos) con un celular detectado en el frame
# para considerar "uso de celular". Decision de ingenieria (analoga a
# distraccion_segundos: la literatura de distraccion al conducir asocia
# el uso del celular con un vistazo sostenido fuera de la tarea de manejar,
# no con una aparicion momentanea), sujeta a calibracion con pruebas reales.
celular_segundos: 2.0

# Tiempo minimo (segundos) entre dos disparos consecutivos del evento, para
# no saturar el registro mientras la condicion persiste. Mismo valor y
# misma justificacion que cooldown_segundos en los demas detectores, por
# consistencia.
cooldown_segundos: 30

# Tiempo maximo (segundos) permitido entre dos frames procesados
# consecutivamente antes de considerar que hubo un "hueco" de deteccion.
# Mismo valor y misma justificacion que gap_maximo_segundos en los demas
# detectores.
gap_maximo_segundos: 1.0
```

```python
CAMPOS_REQUERIDOS_CELULAR = (
    "confianza_umbral",
    "celular_segundos",
    "cooldown_segundos",
    "gap_maximo_segundos",
)


@dataclass
class ConfigCelular:
    confianza_umbral: float
    celular_segundos: float
    cooldown_segundos: float
    gap_maximo_segundos: float


def cargar_config_celular(path: str) -> ConfigCelular: ...  # mismo patron que cargar_config_distraccion
```

### `dsd/db.py` — sin cambios

La tabla `events` (`session_id, tipo, valor, timestamp, synced`) ya es genérica. El nuevo evento se inserta con el `registrar_evento` existente, usando `tipo="uso_celular"`. No se requiere migración de esquema.

### `dsd/main.py` — integración

Nueva llamada a `detectar_objetos(frame)` dentro del bloque `if estado.estado == Estado.ACTIVA:` ya existente, pero **como hermano** del bloque `if landmarks is not None:`, no anidada dentro — es una señal independiente que no depende de que se haya detectado una cara ese frame.

```python
if estado.estado == Estado.ACTIVA:
    landmarks = detectar_landmarks(frame)
    if landmarks is not None:
        ...  # somnolencia, distraccion (sin cambios)

    objetos = detectar_objetos(frame)
    estado_celular, eventos_celular = procesar_objetos(
        estado_celular, objetos, timestamp, config_celular
    )
    for evento_celular in eventos_celular:
        ...  # mismo patron de print + registrar_evento que los demas eventos
```

`estado_celular = estado_inicial_celular()` se inicializa junto a `estado_somnolencia`/`estado_distraccion` al arrancar, y se reinicia en el mismo punto donde ya se reinician esos dos al iniciar una nueva sesión (`sesion_iniciada`).

Ambos modelos (`FaceLandmarker` y `ObjectDetector`) corren **sincrónicamente en el loop principal**, igual que la detección de landmarks hoy — no en el hilo aparte de reconocimiento (ese hilo es específicamente para DeepFace, que es más pesado). Si la verificación manual muestra que el framerate se degrada de forma notoria al sumar esta segunda inferencia por frame, se evalúa mover `detectar_objetos` a su propio hilo en una iteración futura — no se resuelve preventivamente aquí (YAGNI).

## Testing

- **Módulos puros con TDD completo:** `dsd/phone_state.py` (mismo estilo que `distraction_state.py`: sostenimiento, cooldown, reinicio al dejar de detectarse, reinicio tras hueco), extensión de `dsd/config.py` (YAML válido/inválido para `ConfigCelular`).
- **Módulos impuros con verificación manual (cámara real):** `dsd/object_detection.py` (sostener un celular real frente a la cámara y confirmar que aparece `ObjetoDetectado(etiqueta="cell phone", ...)` con confianza razonable), `dsd/main.py` (flujo completo: sostener el celular >2s → evento `uso_celular` impreso y persistido en `events`; cooldown de 30s respetado; confirmar visualmente que el framerate de la ventana no se degrada de forma notoria).

## Fuera de alcance

Cigarro (sub-proyecto futuro, requiere su propia investigación de modelo — no hay clase COCO equivalente), proximidad del celular a la cara/mano, alertas audio/visuales, mover la inferencia de objetos a un hilo aparte (solo si la verificación manual lo justifica), sincronización con la central, puerto a Orange Pi.
