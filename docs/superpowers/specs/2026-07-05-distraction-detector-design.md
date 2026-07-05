# Detector de Distracción (pose de cabeza + mirada) — Diseño

## Contexto

Este es el tercer sub-proyecto del sistema de detección de somnolencia/distracción del conductor. Los dos anteriores ya están completos y pusheados a `https://github.com/cursorcl/deteccion_conductores`:

1. Reconocimiento de conductor + máquina de estados de sesión (`dsd/db.py`, `dsd/session_state.py`, `dsd/enroll.py`, `dsd/recognition.py`, `dsd/main.py`).
2. Detector de somnolencia — microsueño (EAR) + fatiga acumulada (PERCLOS) (`dsd/eye_metrics.py`, `dsd/config.py`, `dsd/drowsiness_state.py`, `dsd/face_mesh.py`, tabla `events` en `dsd/db.py`).

De los 4 detectores de comportamiento planeados (somnolencia, distracción, celular, cigarro), este es el segundo en el orden acordado. Al igual que en somnolencia, el alcance de este sub-proyecto es **solo detección + persistencia**, sin alertas (eso queda para un sub-proyecto futuro compartido con los demás detectores).

## Objetivo

Detectar distracción del conductor combinando dos señales independientes:

1. **Pose de cabeza**: la cabeza gira o se inclina fuera de un rango "mirando al frente" de forma sostenida (p.ej. mirar el radio, un pasajero, o hacia abajo).
2. **Mirada (gaze)**: los ojos se desvían del centro de forma sostenida, incluso sin que la cabeza gire (p.ej. mirar el celular en el regazo sin mover la cabeza).

Cada señal genera su propio tipo de evento, independiente entre sí, con su propio temporizador de sostenimiento y su propio cooldown — mismo patrón conceptual que microsueño/PERCLOS en el detector de somnolencia, pero sin necesidad de una métrica de ventana acumulada (PERCLOS-style), porque la literatura de distracción se basa en la duración de un único vistazo continuo fuera de la carretera, no en una fracción acumulada de tiempo.

Activo solo mientras `dsd.session_state` reporta `Estado.ACTIVA` (mismo gate que somnolencia).

## Arquitectura

```
dsd/face_mesh.py        (MODIFICADO) — una sola llamada a Mediapipe por frame,
                          expone puntos de ojos + iris + matriz de rotación
dsd/head_pose.py         (NUEVO, puro) — matriz de rotación → yaw/pitch en grados
dsd/gaze_metrics.py      (NUEVO, puro) — posición del iris → ratio de mirada
dsd/distraction_state.py (NUEVO, puro) — máquina de estados (2 temporizadores)
dsd/config.py            (MODIFICADO) — + ConfigDistraccion, cargar_config_distraccion
config/distraccion.yaml  (NUEVO) — umbrales documentados
dsd/db.py                (SIN CAMBIOS) — tabla `events` ya es genérica, se reutiliza
dsd/main.py              (MODIFICADO) — integra distracción junto a somnolencia
```

### `dsd/face_mesh.py` — refactor a una sola inferencia por frame

Actualmente `detectar_ojos(frame)` corre `FaceLandmarker.detect()` una vez y devuelve solo los puntos de ojos. Para evitar correr el detector dos veces por frame (una para somnolencia, otra para distracción), se reemplaza por:

```python
@dataclass
class ResultadoLandmarks:
    puntos_ojo_derecho: List[Tuple[float, float]]
    puntos_ojo_izquierdo: List[Tuple[float, float]]
    iris_derecho: Tuple[float, float]
    iris_izquierdo: Tuple[float, float]
    matriz_rotacion: List[List[float]]  # submatriz 3x3 de rotación


def detectar_landmarks(frame) -> Optional[ResultadoLandmarks]:
    ...
```

- Se activa `output_facial_transformation_matrixes=True` en `FaceLandmarkerOptions` (hoy en `False`).
- Los mismos 6 puntos de ojo ya usados para EAR (`INDICES_OJO_DERECHO`/`INDICES_OJO_IZQUIERDO`) sirven también como referencia de "esquinas del ojo" para el ratio de mirada — no se necesitan índices de landmarks adicionales para eso.
- Los puntos de iris ya vienen incluidos en los 478 landmarks de Mediapipe (no requieren configuración adicional, a diferencia de la matriz de rotación).
- `dsd/main.py` se actualiza para usar `detectar_landmarks` en vez de `detectar_ojos`.

### `dsd/head_pose.py` (nuevo, puro)

```python
def calcular_yaw_pitch(matriz_rotacion: Sequence[Sequence[float]]) -> Tuple[float, float]:
    """Extrae (yaw, pitch) en grados desde una matriz de rotacion 3x3,
    usando la descomposicion estandar de angulos de Euler."""
```

Testeado con matrices de rotación sintéticas (construidas a partir de ángulos conocidos), sin dependencia de cámara ni Mediapipe — permite verificar matemáticamente que el ángulo recuperado coincide con el ángulo usado para construir la matriz.

### `dsd/gaze_metrics.py` (nuevo, puro)

```python
def calcular_gaze_ratio(
    iris: Tuple[float, float], puntos_ojo: Sequence[Tuple[float, float]]
) -> Tuple[float, float]:
    """Retorna (ratio_horizontal, ratio_vertical) de la posicion del iris
    dentro del contorno del ojo. 0.5 = centrado; valores que se alejan de
    0.5 indican mirada desviada hacia una esquina/parpado."""
```

Usa los mismos 6 puntos EAR (`esquina_externa`, `parpado_superior_1/2`, `esquina_interna`, `parpado_inferior_2/1`) como referencia geométrica. Testeado con puntos sintéticos.

### `dsd/distraction_state.py` (nuevo, puro)

Mismo patrón que `dsd/drowsiness_state.py`: dos temporizadores de sostenimiento independientes (cabeza girada, mirada desviada), cada uno con su propio cooldown wall-clock, sin resetearse por reaperturas momentáneas más que al volver a mirar al frente (mismo principio que el temporizador de microsueño).

```python
@dataclass
class EstadoDistraccion:
    cabeza_girada_inicio: Optional[float] = None
    mirada_desviada_inicio: Optional[float] = None
    ultimo_disparo_cabeza: Optional[float] = None
    ultimo_disparo_mirada: Optional[float] = None


@dataclass
class EventoDistraccion:
    tipo: str  # "distraccion_cabeza" | "distraccion_mirada"
    valor: float  # duracion del vistazo sostenido, en segundos


def estado_inicial_distraccion() -> EstadoDistraccion: ...


def procesar_pose_y_mirada(
    estado: EstadoDistraccion,
    yaw: float,
    pitch: float,
    gaze_horizontal: float,
    gaze_vertical: float,
    timestamp: float,
    config: ConfigDistraccion,
) -> tuple[EstadoDistraccion, list[EventoDistraccion]]: ...
```

No requiere ventana deslizante ni chequeo de cobertura mínima (a diferencia de PERCLOS) — la condición es puramente "¿cuánto tiempo lleva sostenida la desviación?", igual que microsueño.

### `config/distraccion.yaml` (nuevo) + `dsd/config.py` (extendido)

```yaml
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

`dsd/config.py` gana un segundo dataclass + loader en el mismo archivo (no un archivo nuevo — sigue siendo, en esencia, "cargar YAML de umbrales documentados"):

```python
@dataclass
class ConfigDistraccion:
    distraccion_segundos: float
    yaw_umbral_grados: float
    pitch_umbral_grados: float
    gaze_ratio_umbral: float
    cooldown_segundos: float


def cargar_config_distraccion(path: str) -> ConfigDistraccion: ...
```

### `dsd/db.py` — sin cambios

La tabla `events` (`session_id, tipo, valor, timestamp, synced`) ya es genérica. Los nuevos eventos se insertan con el `registrar_evento` existente, usando `tipo="distraccion_cabeza"` o `tipo="distraccion_mirada"`. No se requiere migración de esquema.

### `dsd/main.py` — integración

Una sola llamada a `detectar_landmarks(frame)` por frame (reemplaza la llamada a `detectar_ojos`), dentro del mismo bloque `if estado.estado == Estado.ACTIVA` ya existente. El resultado alimenta tanto el cálculo de EAR (somnolencia, sin cambios de lógica) como el cálculo de yaw/pitch y gaze ratio (distracción, nuevo). Mismo patrón de impresión en consola + `registrar_evento` condicionado a `session_id_activo is not None` ya usado para eventos de somnolencia.

## Testing

- **Módulos puros con TDD completo:** `dsd/head_pose.py` (matrices sintéticas), `dsd/gaze_metrics.py` (puntos sintéticos), `dsd/distraction_state.py` (mismo estilo de tests que `drowsiness_state.py`: sostenimiento, cooldown, reinicio al mirar al frente), extensión de `dsd/config.py` (YAML válido/inválido).
- **Módulos impuros con verificación manual (cámara real):** `dsd/face_mesh.py` (girar la cabeza y confirmar que yaw/pitch cambian de forma coherente; mover los ojos sin girar la cabeza y confirmar que el ratio de mirada cambia), `dsd/main.py` (flujo completo: sesión activa, girar la cabeza >2s → evento `distraccion_cabeza`, mirar al costado sin girar la cabeza >2s → evento `distraccion_mirada`, cooldown de 30s respetado, eventos persistidos correctamente en `events` con el `tipo` correcto).

## Fuera de alcance

Alertas audio/visuales, sincronización con la central (la columna `synced` ya existe y se reutiliza igual que en somnolencia), detectores de celular/cigarro, puerto a Orange Pi.
