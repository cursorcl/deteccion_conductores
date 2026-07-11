# Detección de Bostezo — Diseño

## Contexto

Extiende el detector de somnolencia ya existente (`dsd/eye_metrics.py`, `dsd/drowsiness_state.py`, `dsd/config.py`, `config/somnolencia.yaml`), que hoy cubre microsueño (EAR) y fatiga acumulada (PERCLOS). El bostezo es una tercera señal de somnolencia bien establecida en la literatura de fatiga al conducir, complementaria a las dos anteriores.

Los índices de landmarks de boca necesarios (topología de 468/478 puntos de Mediapipe Face Mesh) se verificaron manualmente contra cámara real antes de fijarlos aquí, siguiendo el mismo proceso ya usado para los índices de iris (ver comentario en `dsd/face_mesh.py`).

Igual que en los detectores existentes, el alcance es **solo detección + persistencia**, sin alertas. Activo solo mientras `dsd.session_state` reporta `Estado.ACTIVA` (mismo gate que el resto).

## Objetivo

Detectar bostezos combinando dos señales relacionadas:

1. **Bostezo individual** (`bostezo`): boca abierta (MAR por encima de un umbral) de forma sostenida durante al menos `bostezo_min_segundos`. La duración mínima existe para distinguir un bostezo real de hablar o gesticular. Mismo patrón que microsueño: temporizador de sostenimiento compartido (`dsd/sustained_timer.py`), dispara una vez por bostezo (cooldown evita contar el mismo bostezo varias veces mientras la boca sigue abierta).
2. **Fatiga por frecuencia de bostezos** (`fatiga_bostezos`): cuando ocurren `bostezo_umbral_cantidad` o más bostezos individuales dentro de una ventana deslizante de `bostezo_ventana_segundos`. Mismo principio que PERCLOS (ventana deslizante + cooldown) pero contando ocurrencias discretas en vez de tiempo ponderado, porque la frecuencia de bostezos (no la duración de cada uno) es el indicador de fatiga acumulada en la literatura.

Ambos eventos se persisten (igual que microsueño y PERCLOS se persisten ambos hoy), dando visibilidad completa del detalle individual y de la tendencia agregada.

## Arquitectura

```
dsd/face_mesh.py         (MODIFICADO) — + puntos_boca en ResultadoLandmarks
dsd/mouth_metrics.py     (NUEVO, puro) — calcular_mar
dsd/drowsiness_state.py  (MODIFICADO) — + temporizador de bostezo + ventana de frecuencia
dsd/config.py            (MODIFICADO) — + campos de bostezo en ConfigSomnolencia
config/somnolencia.yaml  (MODIFICADO) — + umbrales de bostezo, documentados
dsd/db.py                (SIN CAMBIOS) — tabla `events` ya es genérica, se reutiliza
dsd/main.py              (MODIFICADO) — calcula MAR e integra con el detector de somnolencia
```

### `dsd/face_mesh.py` — índices de boca verificados

```python
# Indices verificados manualmente contra camara real (mismo proceso que
# INDICE_IRIS_DERECHO/IZQUIERDO) para el calculo de MAR (Mouth Aspect
# Ratio), en el mismo orden de 6 puntos que EAR: [comisura_izquierda,
# labio_superior_izquierdo, labio_superior_derecho, comisura_derecha,
# labio_inferior_derecho, labio_inferior_izquierdo].
INDICES_BOCA = [61, 40, 270, 291, 314, 84]
```

`ResultadoLandmarks` gana el campo `puntos_boca: List[Tuple[float, float]]`, poblado en `detectar_landmarks` igual que los puntos de ojo.

### `dsd/mouth_metrics.py` (nuevo, puro)

```python
def calcular_mar(puntos_boca: Sequence[Tuple[float, float]]) -> float:
    """Calcula el Mouth Aspect Ratio (MAR), geometricamente identico al
    EAR (ver dsd/eye_metrics.py) pero aplicado a los 6 puntos de boca:
    (comisura_izquierda, labio_superior_izquierdo, labio_superior_derecho,
    comisura_derecha, labio_inferior_derecho, labio_inferior_izquierdo).

    MAR = (dist(p2, p6) + dist(p3, p5)) / (2 * dist(p1, p4))

    Boca cerrada -> MAR bajo; boca abierta (bostezo) -> MAR alto (a
    diferencia del EAR, donde el ojo cerrado da el valor bajo)."""
```

Misma validación de 6 puntos y mismo manejo de división por cero que `calcular_ear`.

### `dsd/drowsiness_state.py` — extensión

```python
@dataclass
class EstadoSomnolencia:
    muestras: List[Muestra] = field(default_factory=list)
    cierre_inicio: Optional[float] = None
    ultimo_disparo_microsueno: Optional[float] = None
    ultimo_disparo_perclos: Optional[float] = None
    primer_timestamp: Optional[float] = None
    ultimo_procesado: Optional[float] = None
    # --- nuevo, bostezo ---
    boca_abierta_inicio: Optional[float] = None
    ultimo_disparo_bostezo: Optional[float] = None
    bostezos: List[float] = field(default_factory=list)  # timestamps
    ultimo_disparo_fatiga_bostezos: Optional[float] = None
```

`procesar_ear` se renombra a **`procesar_somnolencia`** (recibe `ear` y `mar`, ya no solo `ear`) — mismo criterio de nombres que `procesar_pose_y_mirada` en `distraction_state.py`, que ya nombra la función por las señales que procesa, no por una sola métrica.

```python
def procesar_somnolencia(
    estado: EstadoSomnolencia,
    ear: float,
    mar: float,
    timestamp: float,
    config: ConfigSomnolencia,
) -> tuple[EstadoSomnolencia, list[EventoSomnolencia]]: ...
```

Lógica nueva, agregada al mismo `hubo_hueco` ya calculado para microsueño/PERCLOS (se reutiliza, no se duplica):

- `boca_abierta = mar > config.mar_umbral`
- Temporizador de sostenimiento (`procesar_temporizador_sostenido`, igual que microsueño) con `umbral_segundos=config.bostezo_min_segundos` y `cooldown_segundos=config.cooldown_segundos` (mismo campo reutilizado, no uno nuevo). Este cooldown cumple doble función: evita contar el mismo bostezo dos veces mientras la boca sigue abierta, y separa bostezos reales consecutivos (con el valor por defecto de 30s, dos bostezos distintos deben estar separados por más de 30s para contarse por separado — igual de razonable que el resto de los cooldowns del sistema, que ya usan este mismo valor). Cada disparo agrega el evento `bostezo` y agrega su timestamp a `estado.bostezos`.
- `estado.bostezos` se recorta a los que caen dentro de `timestamp - config.bostezo_ventana_segundos` (mismo patrón de recorte que `muestras` en PERCLOS).
- Si `len(bostezos_en_ventana) >= config.bostezo_umbral_cantidad` y no está en cooldown (`estado.ultimo_disparo_fatiga_bostezos`, comparado contra `config.cooldown_segundos`, mismo campo — mismo patrón que `ultimo_disparo_perclos`), dispara `fatiga_bostezos` con `valor=len(bostezos_en_ventana)`.

### `dsd/config.py` + `config/somnolencia.yaml` — campos nuevos

```python
@dataclass
class ConfigSomnolencia:
    # ... campos existentes ...
    mar_umbral: float
    bostezo_min_segundos: float
    bostezo_ventana_segundos: float
    bostezo_umbral_cantidad: float
```

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

Reutiliza `cooldown_segundos` y `gap_maximo_segundos` ya existentes en el mismo archivo — no se duplican.

### `dsd/db.py` — sin cambios

Igual que en somnolencia/distracción: la tabla `events` genérica ya soporta `tipo="bostezo"` y `tipo="fatiga_bostezos"` sin migración de esquema.

### `dsd/main.py` — integración

`detectar_landmarks(frame)` ya devuelve `puntos_boca` (viene del cambio en `face_mesh.py`). Se agrega:

```python
mar = calcular_mar(landmarks.puntos_boca)
estado_somnolencia, eventos_somnolencia = procesar_somnolencia(
    estado_somnolencia, ear_promedio, mar, timestamp, config_somnolencia
)
```

reemplazando la llamada actual a `procesar_ear`. Mismo patrón de impresión en consola + `registrar_evento` condicionado a `session_id_activo is not None` ya usado para los demás eventos de somnolencia — sin lógica nueva de persistencia.

## Testing

- **Módulos puros con TDD completo:** `dsd/mouth_metrics.py` (puntos sintéticos, boca cerrada vs abierta, división por cero), extensión de `dsd/drowsiness_state.py` (bostezo individual sostenido, no-doble-conteo del mismo bostezo por cooldown, frecuencia dentro/fuera de ventana, cooldown de `fatiga_bostezos`, reinicio del temporizador de bostezo tras un hueco de detección — mismo estilo que los tests ya existentes de microsueño/PERCLOS), extensión de `tests/test_config.py` (nuevos campos requeridos).
- **Módulos impuros con verificación manual (cámara real):** `dsd/face_mesh.py` (confirmar que `puntos_boca` sigue los labios al abrir/cerrar la boca), `dsd/main.py` (flujo completo: bostezar >1.5s → evento `bostezo`; repetir hasta alcanzar `bostezo_umbral_cantidad` dentro de la ventana → evento `fatiga_bostezos`; cooldown respetado; eventos persistidos con el `tipo` correcto en `events`).

## Fuera de alcance

Alertas audio/visuales, calibración final de `mar_umbral`/duraciones/ventana con datos reales (quedan marcados como valores iniciales de ingeniería), detectores de celular/cigarro, sincronización con la central, puerto a Orange Pi.
