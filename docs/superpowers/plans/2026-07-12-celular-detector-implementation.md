# Detector de Uso de Celular Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detectar "uso de celular" (etiqueta `"cell phone"` sostenida en el frame, vía un modelo de detección de objetos) y persistir el evento `uso_celular`, con el mismo patrón de temporizador sostenido + cooldown que microsueño/distracción/bostezo.

**Architecture:** Un modelo `ObjectDetector` de Mediapipe (EfficientDet-Lite0 pre-entrenado sobre COCO, ya trae la clase `"cell phone"`) envuelto en `dsd/object_detection.py` (análogo a `dsd/face_mesh.py`); `dsd/phone_state.py` reutiliza `dsd/sustained_timer.py` sin cambios (mismo patrón de un único temporizador que ya usa `distraction_state.py`); `dsd/main.py` corre la detección de objetos como señal independiente de la detección facial (no depende de que haya landmarks ese frame).

**Tech Stack:** Python 3.11, pytest, PyYAML, Mediapipe Tasks API (`vision.ObjectDetector`), OpenCV.

## Global Constraints

- Modelo: `models/efficientdet_lite0.tflite`, descargado de `https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite` (URL oficial de Google/Mediapipe, HTTP 200 confirmado en la sesión de diseño). Asset binario de terceros — **no se versiona en git** (`models/` ya está en `.gitignore`).
- Etiqueta objetivo: `"cell phone"` (nombre exacto de categoría devuelto por el modelo para la clase COCO correspondiente).
- "Uso de celular" = etiqueta `"cell phone"` en **cualquier parte del frame** (no se exige proximidad a la cara) con confianza ≥ `confianza_umbral`, sostenida ≥ `celular_segundos`.
- Reutilizar `dsd/sustained_timer.py` (`EstadoTemporizadorSostenido`, `procesar_temporizador_sostenido`) — no crear un temporizador nuevo.
- El umbral de confianza real (`confianza_umbral`) vive en `dsd/phone_state.py` vía config, NO en `dsd/object_detection.py` — ese módulo solo aplica un piso bajo fijo (`score_threshold=0.2`) a nivel del modelo para no acumular ruido, igual que `ear_umbral` vive en `drowsiness_state.py` y no en `face_mesh.py`.
- La detección de objetos corre como señal independiente de la detección facial: bajo `Estado.ACTIVA`, pero **no anidada** dentro de `if landmarks is not None:`.
- `dsd/db.py` no se modifica — la tabla `events` ya es genérica.
- Sin alertas audio/visuales — solo detección + persistencia.
- Cigarro está fuera de alcance de este plan.

---

## Task 1: `dsd/object_detection.py` — wrapper de detección de objetos

**Files:**
- Create: `dsd/object_detection.py`

**Interfaces:**
- Produces: `ObjetoDetectado` (dataclass: `etiqueta: str`, `confianza: float`); `detectar_objetos(frame) -> List[ObjetoDetectado]`.

No hay test automatizado para este archivo (depende de cámara/Mediapipe real — mismo criterio ya aplicado a `dsd/face_mesh.py`, sin `tests/test_face_mesh.py` en el proyecto). Verificación manual en Step 4.

- [ ] **Step 1: Descargar el modelo**

```bash
curl -sL -o models/efficientdet_lite0.tflite "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite"
ls -la models/efficientdet_lite0.tflite
```
Expected: el archivo existe y pesa aproximadamente 13.8MB (ya descargado y verificado funcional en la sesión de diseño — si el archivo ya existe con ese tamaño, el `curl` es idempotente y no hace falta repetirlo).

- [ ] **Step 2: Crear `dsd/object_detection.py`**

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

- [ ] **Step 3: Confirmar que el resto de la suite sigue pasando**

Run: `pytest -v`
Expected: todos los tests existentes PASS (este archivo no lo importa nada más todavía).

- [ ] **Step 4: Verificación manual (cámara real)**

```bash
source .venv/bin/activate
python -c "
import cv2
from dsd.object_detection import detectar_objetos

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()
if not ret:
    print('no se pudo leer frame de la camara')
else:
    objetos = detectar_objetos(frame)
    for o in objetos:
        print(o.etiqueta, round(o.confianza, 2))
"
```
Sosteniendo un celular real frente a la cámara: debe imprimirse una línea `cell phone <confianza>` con confianza razonable (>0.3 aprox). Sin ningún objeto reconocible en cuadro, la lista puede salir vacía — eso no es un fallo del código, es esperado.

- [ ] **Step 5: Commit**

```bash
git add dsd/object_detection.py
git commit -m "feat: agregar wrapper de deteccion de objetos (Mediapipe ObjectDetector)"
```

Nota: `models/efficientdet_lite0.tflite` NO se agrega (está en `.gitignore`, igual que `models/face_landmarker.task`).

---

## Task 2: `dsd/config.py` + `config/celular.yaml` — umbrales de celular

**Files:**
- Modify: `dsd/config.py`
- Create: `config/celular.yaml`
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces: `ConfigCelular` (dataclass: `confianza_umbral: float`, `celular_segundos: float`, `cooldown_segundos: float`, `gap_maximo_segundos: float`), `cargar_config_celular(path: str) -> ConfigCelular`.

- [ ] **Step 1: Escribir el test que falla**

Agregar al final de `tests/test_config.py`:

```python


from dsd.config import ConfigCelular, cargar_config_celular

YAML_VALIDO_CELULAR = """
confianza_umbral: 0.5
celular_segundos: 2.0
cooldown_segundos: 30
gap_maximo_segundos: 1.0
"""


def test_cargar_config_celular_retorna_los_valores_correctos(tmp_path):
    ruta = tmp_path / "celular.yaml"
    ruta.write_text(YAML_VALIDO_CELULAR)

    config = cargar_config_celular(str(ruta))

    assert config == ConfigCelular(
        confianza_umbral=0.5,
        celular_segundos=2.0,
        cooldown_segundos=30.0,
        gap_maximo_segundos=1.0,
    )


def test_cargar_config_celular_convierte_valores_a_float(tmp_path):
    ruta = tmp_path / "celular.yaml"
    ruta.write_text(YAML_VALIDO_CELULAR)

    config = cargar_config_celular(str(ruta))

    assert isinstance(config.confianza_umbral, float)
    assert isinstance(config.cooldown_segundos, float)


def test_cargar_config_celular_clave_faltante_lanza_keyerror(tmp_path):
    ruta = tmp_path / "celular.yaml"
    ruta.write_text("confianza_umbral: 0.5\n")

    with pytest.raises(KeyError):
        cargar_config_celular(str(ruta))


def test_cargar_config_celular_archivo_real_del_proyecto():
    config = cargar_config_celular("config/celular.yaml")
    assert config.confianza_umbral == 0.5
    assert config.celular_segundos == 2.0
    assert config.cooldown_segundos == 30.0
    assert config.gap_maximo_segundos == 1.0
```

- [ ] **Step 2: Confirmar que falla**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ImportError: cannot import name 'ConfigCelular' from 'dsd.config'`

- [ ] **Step 3: Agregar `ConfigCelular` a `dsd/config.py`**

Agregar al final del archivo (después de `cargar_config_distraccion`):

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


def cargar_config_celular(path: str) -> ConfigCelular:
    with open(path, "r", encoding="utf-8") as archivo:
        datos = yaml.safe_load(archivo)

    faltantes = [campo for campo in CAMPOS_REQUERIDOS_CELULAR if campo not in datos]
    if faltantes:
        raise KeyError(
            f"Faltan claves requeridas en el archivo de configuracion '{path}': {faltantes}"
        )

    return ConfigCelular(
        confianza_umbral=float(datos["confianza_umbral"]),
        celular_segundos=float(datos["celular_segundos"]),
        cooldown_segundos=float(datos["cooldown_segundos"]),
        gap_maximo_segundos=float(datos["gap_maximo_segundos"]),
    )
```

- [ ] **Step 4: Crear `config/celular.yaml`**

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

- [ ] **Step 5: Confirmar que pasa**

Run: `pytest tests/test_config.py -v`
Expected: todos los tests PASS

- [ ] **Step 6: Confirmar que el resto de la suite sigue pasando**

Run: `pytest -v`
Expected: todos PASS

- [ ] **Step 7: Commit**

```bash
git add dsd/config.py config/celular.yaml tests/test_config.py
git commit -m "feat: agregar ConfigCelular y config/celular.yaml"
```

---

## Task 3: `dsd/phone_state.py` — temporizador sostenido de uso de celular

**Files:**
- Create: `dsd/phone_state.py`
- Test: `tests/test_phone_state.py`

**Interfaces:**
- Consumes: `ObjetoDetectado` (Task 1), `ConfigCelular` (Task 2), `EstadoTemporizadorSostenido`/`procesar_temporizador_sostenido` de `dsd/sustained_timer.py` (sin cambios).
- Produces: `EstadoCelular` (dataclass: `deteccion_inicio: Optional[float] = None`, `ultimo_disparo: Optional[float] = None`, `ultimo_procesado: Optional[float] = None`), `EventoCelular` (dataclass: `tipo: str`, `valor: float`), `estado_inicial_celular() -> EstadoCelular`, `procesar_objetos(estado, objetos, timestamp, config) -> tuple[EstadoCelular, list[EventoCelular]]`.

- [ ] **Step 1: Escribir el test que falla**

Crear `tests/test_phone_state.py`:

```python
from dsd.config import ConfigCelular
from dsd.object_detection import ObjetoDetectado
from dsd.phone_state import EventoCelular, estado_inicial_celular, procesar_objetos

CONFIG = ConfigCelular(
    confianza_umbral=0.5,
    celular_segundos=2.0,
    cooldown_segundos=30.0,
    gap_maximo_segundos=1.0,
)

CELULAR_CONFIANZA_ALTA = [ObjetoDetectado(etiqueta="cell phone", confianza=0.8)]
CELULAR_CONFIANZA_BAJA = [ObjetoDetectado(etiqueta="cell phone", confianza=0.3)]
OTRO_OBJETO = [ObjetoDetectado(etiqueta="person", confianza=0.9)]
SIN_OBJETOS = []


def test_estado_inicial_no_tiene_temporizador_activo():
    estado = estado_inicial_celular()
    assert estado.deteccion_inicio is None
    assert estado.ultimo_disparo is None
    assert estado.ultimo_procesado is None


def test_sin_objetos_no_acumula_deteccion():
    estado = estado_inicial_celular()
    nuevo_estado, eventos = procesar_objetos(estado, SIN_OBJETOS, timestamp=0.0, config=CONFIG)
    assert eventos == []
    assert nuevo_estado.deteccion_inicio is None


def test_otro_objeto_no_cuenta_como_celular():
    estado = estado_inicial_celular()
    nuevo_estado, eventos = procesar_objetos(estado, OTRO_OBJETO, timestamp=0.0, config=CONFIG)
    assert eventos == []
    assert nuevo_estado.deteccion_inicio is None


def test_confianza_baja_no_cuenta_como_celular_detectado():
    estado = estado_inicial_celular()
    nuevo_estado, eventos = procesar_objetos(
        estado, CELULAR_CONFIANZA_BAJA, timestamp=0.0, config=CONFIG
    )
    assert eventos == []
    assert nuevo_estado.deteccion_inicio is None


def test_deteccion_breve_no_dispara_uso_celular():
    estado = estado_inicial_celular()
    for t in [0.0, 1.0]:
        estado, eventos = procesar_objetos(
            estado, CELULAR_CONFIANZA_ALTA, timestamp=t, config=CONFIG
        )
        assert eventos == []


def test_deteccion_exactamente_en_el_limite_no_dispara():
    # Pasos densos (<= gap_maximo_segundos) para que el chequeo de hueco no
    # interfiera con la condicion de limite que este test quiere ejercitar.
    estado = estado_inicial_celular()
    for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
        estado, eventos = procesar_objetos(
            estado, CELULAR_CONFIANZA_ALTA, timestamp=t, config=CONFIG
        )
    assert eventos == []


def test_deteccion_sostenida_dispara_uso_celular():
    estado = estado_inicial_celular()
    for t in [0.0, 1.0, 2.0]:
        estado, _ = procesar_objetos(
            estado, CELULAR_CONFIANZA_ALTA, timestamp=t, config=CONFIG
        )
    estado, eventos = procesar_objetos(
        estado, CELULAR_CONFIANZA_ALTA, timestamp=2.1, config=CONFIG
    )
    assert eventos == [EventoCelular(tipo="uso_celular", valor=2.1)]


def test_dejar_de_detectar_reinicia_temporizador():
    estado = estado_inicial_celular()
    estado, _ = procesar_objetos(estado, CELULAR_CONFIANZA_ALTA, timestamp=0.0, config=CONFIG)
    estado, _ = procesar_objetos(estado, SIN_OBJETOS, timestamp=0.5, config=CONFIG)
    assert estado.deteccion_inicio is None
    estado, _ = procesar_objetos(estado, CELULAR_CONFIANZA_ALTA, timestamp=0.6, config=CONFIG)
    assert estado.deteccion_inicio == 0.6


def test_uso_celular_no_re_dispara_dentro_del_cooldown():
    # Muestreo continuo (paso 0.5s, bajo gap_maximo_segundos) para simular
    # celular detectado de forma ininterrumpida durante todo el cooldown.
    estado = estado_inicial_celular()
    eventos_celular = []
    t = 0.0
    while t <= 31.5:
        estado, eventos = procesar_objetos(
            estado, CELULAR_CONFIANZA_ALTA, timestamp=t, config=CONFIG
        )
        eventos_celular += [e for e in eventos if e.tipo == "uso_celular"]
        t += 0.5
    assert len(eventos_celular) == 1
    assert eventos_celular[0].valor == 2.5


def test_uso_celular_re_dispara_tras_cooldown_si_sigue_detectado():
    estado = estado_inicial_celular()
    eventos_celular = []
    t = 0.0
    while t <= 33.0:
        estado, eventos = procesar_objetos(
            estado, CELULAR_CONFIANZA_ALTA, timestamp=t, config=CONFIG
        )
        eventos_celular += [e for e in eventos if e.tipo == "uso_celular"]
        t += 0.5
    assert len(eventos_celular) == 2
    assert eventos_celular[0].valor == 2.5
    assert eventos_celular[1].valor == 32.5


def test_multiples_objetos_uno_es_celular_con_confianza_suficiente():
    estado = estado_inicial_celular()
    objetos = [
        ObjetoDetectado(etiqueta="person", confianza=0.9),
        ObjetoDetectado(etiqueta="cell phone", confianza=0.6),
    ]
    nuevo_estado, _ = procesar_objetos(estado, objetos, timestamp=0.0, config=CONFIG)
    assert nuevo_estado.deteccion_inicio == 0.0


def test_uso_celular_no_dispara_con_valor_inflado_tras_hueco_prolongado():
    # Reproduce el mismo hallazgo ya aplicado a microsueno/distraccion/
    # bostezo: si el modelo no proceso frames durante un tramo largo y
    # luego se retoma con el celular ya detectado, el temporizador NO debe
    # asumir que estuvo detectado desde antes del hueco.
    estado = estado_inicial_celular()
    estado, _ = procesar_objetos(estado, SIN_OBJETOS, timestamp=0.0, config=CONFIG)
    # Hueco prolongado: sin llamadas entre t=0 y t=50.
    estado, eventos = procesar_objetos(
        estado, CELULAR_CONFIANZA_ALTA, timestamp=50.0, config=CONFIG
    )
    assert eventos == []
    assert estado.deteccion_inicio == 50.0
```

- [ ] **Step 2: Confirmar que falla**

Run: `pytest tests/test_phone_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dsd.phone_state'`

- [ ] **Step 3: Implementar `dsd/phone_state.py`**

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

    # Un hueco (tiempo excesivo desde el ultimo frame procesado) invalida
    # el supuesto de que la condicion se mantuvo continuamente durante ese
    # tramo -- ver dsd/sustained_timer.py.
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

- [ ] **Step 4: Confirmar que pasa**

Run: `pytest tests/test_phone_state.py -v`
Expected: todos los tests PASS

- [ ] **Step 5: Confirmar que el resto de la suite sigue pasando**

Run: `pytest -v`
Expected: todos PASS

- [ ] **Step 6: Commit**

```bash
git add dsd/phone_state.py tests/test_phone_state.py
git commit -m "feat: agregar deteccion de uso de celular (phone_state)"
```

---

## Task 4: `dsd/main.py` — integración

**Files:**
- Modify: `dsd/main.py`

**Interfaces:**
- Consumes: `detectar_objetos` (Task 1), `ConfigCelular`/`cargar_config_celular` (Task 2), `estado_inicial_celular`/`procesar_objetos`/`EventoCelular` (Task 3).

No hay test automatizado para `dsd/main.py` (mismo criterio que el resto del archivo). Verificación manual en Step 6.

- [ ] **Step 1: Agregar los imports**

Reemplazar:

```python
from dsd.config import cargar_config, cargar_config_distraccion
```

por:

```python
from dsd.config import cargar_config, cargar_config_celular, cargar_config_distraccion
```

Agregar junto a los demás imports de un solo nombre de `dsd.*` (después de `from dsd.mouth_metrics import calcular_mar`):

```python
from dsd.object_detection import detectar_objetos
from dsd.phone_state import estado_inicial_celular, procesar_objetos
```

- [ ] **Step 2: Agregar la ruta de config**

Reemplazar:

```python
RUTA_CONFIG_SOMNOLENCIA = "config/somnolencia.yaml"
RUTA_CONFIG_DISTRACCION = "config/distraccion.yaml"
```

por:

```python
RUTA_CONFIG_SOMNOLENCIA = "config/somnolencia.yaml"
RUTA_CONFIG_DISTRACCION = "config/distraccion.yaml"
RUTA_CONFIG_CELULAR = "config/celular.yaml"
```

- [ ] **Step 3: Cargar la config y el estado inicial en `main()`**

Reemplazar:

```python
    conn = init_db(RUTA_DB)
    config_somnolencia = cargar_config(RUTA_CONFIG_SOMNOLENCIA)
    config_distraccion = cargar_config_distraccion(RUTA_CONFIG_DISTRACCION)
```

por:

```python
    conn = init_db(RUTA_DB)
    config_somnolencia = cargar_config(RUTA_CONFIG_SOMNOLENCIA)
    config_distraccion = cargar_config_distraccion(RUTA_CONFIG_DISTRACCION)
    config_celular = cargar_config_celular(RUTA_CONFIG_CELULAR)
```

Reemplazar:

```python
    estado = estado_inicial()
    estado_somnolencia = estado_inicial_somnolencia()
    estado_distraccion = estado_inicial_distraccion()
    session_id_activo = None
```

por:

```python
    estado = estado_inicial()
    estado_somnolencia = estado_inicial_somnolencia()
    estado_distraccion = estado_inicial_distraccion()
    estado_celular = estado_inicial_celular()
    session_id_activo = None
```

- [ ] **Step 4: Reiniciar `estado_celular` al iniciar sesión**

Reemplazar:

```python
                    # Reinicia el rastreo de somnolencia y distraccion: cada
                    # sesion (mismo conductor u otro) empieza con los
                    # temporizadores en blanco.
                    estado_somnolencia = estado_inicial_somnolencia()
                    estado_distraccion = estado_inicial_distraccion()
```

por:

```python
                    # Reinicia el rastreo de somnolencia, distraccion y
                    # celular: cada sesion (mismo conductor u otro) empieza
                    # con los temporizadores en blanco.
                    estado_somnolencia = estado_inicial_somnolencia()
                    estado_distraccion = estado_inicial_distraccion()
                    estado_celular = estado_inicial_celular()
```

- [ ] **Step 5: Agregar el bloque de detección de celular**

Este bloque va **como hermano** de `if landmarks is not None:` (mismo nivel de indentación, no anidado dentro), pero sigue dentro de `if estado.estado == Estado.ACTIVA:`. Reemplazar el cierre del bloque de distracción y lo que sigue:

```python
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
```

por:

```python
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

                objetos = detectar_objetos(frame)
                estado_celular, eventos_celular = procesar_objetos(
                    estado_celular, objetos, timestamp, config_celular
                )
                for evento_celular in eventos_celular:
                    ahora_iso = datetime.now(timezone.utc).isoformat()
                    print(
                        f"Evento de celular: {evento_celular.tipo} "
                        f"(valor={evento_celular.valor:.3f})"
                    )
                    if session_id_activo is not None:
                        registrar_evento(
                            conn,
                            session_id_activo,
                            evento_celular.tipo,
                            evento_celular.valor,
                            ahora_iso,
                        )
                    else:
                        print("Advertencia: evento de celular no persistido, no hay sesion activa en la base de datos.")

            if estado.estado == Estado.ACTIVA:
                texto = f"Sesion activa: {estado.conductor_actual}"
```

Notar la desindentación: `objetos = detectar_objetos(frame)` queda al mismo nivel que `landmarks = detectar_landmarks(frame)` (dentro de `if estado.estado == Estado.ACTIVA:`, fuera de `if landmarks is not None:`), no dentro del bloque de distracción.

- [ ] **Step 6: Confirmar que la suite completa sigue pasando**

Run: `pytest -v`
Expected: todos los tests PASS

- [ ] **Step 7: Verificación manual (cámara real)**

```bash
source .venv/bin/activate
python -m dsd.main
```

Con una sesión activa (conductor ya enrolado y reconocido):
1. Sostener un celular real frente a la cámara durante más de 2 segundos → debe imprimirse `Evento de celular: uso_celular (valor=X.XXX)`.
2. Bajar el celular, esperar, repetir → el cooldown de 30s debe respetarse (no debe imprimirse un segundo evento antes de esos 30s).
3. Observar si el framerate de la ventana se degrada de forma notoria al agregar esta segunda inferencia por frame (además de la de landmarks faciales) — si se nota lento, es una señal para una futura iteración que mueva `detectar_objetos` a un hilo aparte (no se resuelve en este plan).
4. Salir con `q` y verificar en `data/app.db` que el evento quedó persistido:

```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/app.db')
for row in conn.execute(\"SELECT tipo, valor, timestamp FROM events WHERE tipo = 'uso_celular' ORDER BY id DESC LIMIT 5\"):
    print(row)
"
```

- [ ] **Step 8: Commit**

```bash
git add dsd/main.py
git commit -m "feat: integrar deteccion de uso de celular en el loop principal"
```

---

## Self-Review

**Cobertura de la spec:**
- Modelo `efficientdet_lite0.tflite` descargado de la URL oficial, gitignored → Task 1. ✓
- `dsd/object_detection.py` (`ObjetoDetectado`, `detectar_objetos`), `score_threshold` bajo a nivel de modelo, decision real en config → Task 1. ✓
- `ConfigCelular` + `config/celular.yaml` con los 4 campos documentados → Task 2. ✓
- `dsd/phone_state.py` reutilizando `sustained_timer.py`, evento `uso_celular` → Task 3. ✓
- "Cualquier parte del frame" (sin proximidad a cara) → reflejado en `phone_state.py` (no usa landmarks). ✓
- Detección de celular como señal independiente de landmarks faciales (no anidada) → Task 4, Step 5. ✓
- Reinicio de `estado_celular` al iniciar sesión → Task 4, Step 4. ✓
- Persistencia vía `registrar_evento` genérico, sin cambios a `dsd/db.py` → confirmado, ningún task lo toca. ✓
- Verificación manual de framerate → Task 4, Step 7. ✓
- Fuera de alcance (cigarro, proximidad a cara/mano, alertas, hilo aparte salvo que se justifique, sync, Orange Pi) → ningún task los incluye. ✓

Sin gaps encontrados.
