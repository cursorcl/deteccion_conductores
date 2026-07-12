# Visualización de Malla Facial de Debug — Diseño

## Contexto

Herramienta de calibración/depuración para `dsd/main.py`: hasta ahora, verificar que los landmarks de Mediapipe caen donde corresponde (ojos, iris, boca) requería scripts sueltos y descartables (como el usado para elegir los índices de boca del detector de bostezo). Este sub-proyecto lleva esa capacidad a la app principal, detrás de un flag de línea de comandos, para no tener que rehacer un script ad-hoc cada vez que se necesite inspeccionar visualmente los landmarks.

Alcance: solo visualización de debug, activada explícitamente por flag. No cambia ninguna lógica de detección (somnolencia, distracción, reconocimiento) ni el comportamiento por defecto de la app.

## Objetivo

Con `python -m dsd.main --malla`, además del comportamiento normal de la app, se dibuja sobre el frame:

1. **Malla completa**: los 478 landmarks crudos de Mediapipe como puntos grises, conectados por líneas finas siguiendo la topología real de triangulación de Mediapipe (`FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION`, ya incluida en la librería instalada — no se hardcodea ninguna tabla de conexiones propia).
2. **Puntos de control**: los puntos que la app ya usa para detección (ojos, iris, boca — los mismos que hoy se computan cada frame para EAR/MAR/gaze/pose) resaltados en amarillo, para distinguirlos visualmente del resto de la malla.

Sin el flag, el comportamiento es idéntico al actual (solo se dibujan los puntos de ojo en amarillo, como hoy).

Igual que el resto de la detección de landmarks, esto solo corre dentro del mismo bloque `Estado.ACTIVA` ya existente (sesión con conductor reconocido) — no se activa la detección de rostro fuera de ese caso.

## Arquitectura

```
dsd/face_mesh.py    (MODIFICADO) — + puntos_todos (478 landmarks crudos) en ResultadoLandmarks
dsd/debug_draw.py   (NUEVO) — dibuja malla completa + puntos de control resaltados
dsd/main.py         (MODIFICADO) — flag --malla (argparse), llama a debug_draw cuando esta activo
```

### `dsd/face_mesh.py` — puntos crudos completos

`ResultadoLandmarks` gana un campo:

```python
puntos_todos: List[Tuple[float, float]]  # los 478 landmarks, en orden (indice 0..477)
```

Poblado en `detectar_landmarks` con el mismo helper `punto()` ya existente, iterando sobre todos los landmarks devueltos por Mediapipe (`[punto(i) for i in range(len(landmarks))]`). No agrega una segunda inferencia — es el mismo resultado de `_detector.detect()` que ya se usa para el resto de los puntos, solo que ahora se expone completo además del subconjunto curado.

### `dsd/debug_draw.py` (nuevo)

```python
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

CONEXIONES_TESELACION = FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION

COLOR_MALLA = (120, 120, 120)   # gris, lineas y puntos de fondo
COLOR_CONTROL = (0, 255, 255)   # amarillo, puntos de control (mismo color que ya usan los ojos hoy)


def dibujar_malla_debug(frame, landmarks: "ResultadoLandmarks") -> None:
    """Dibuja sobre `frame` (in-place, mismo patron que cv2.circle/cv2.putText
    ya usados en dsd/main.py): la malla completa de 478 puntos + lineas de
    teselacion en gris, y los puntos de control que la app usa para deteccion
    (ojos, iris, boca) resaltados en amarillo."""
```

Implementación:
- Para cada `Connection(start, end)` en `CONEXIONES_TESELACION`: `cv2.line` entre `puntos_todos[start]` y `puntos_todos[end]`, color gris, grosor 1.
- Para cada punto en `puntos_todos`: `cv2.circle` radio 1, color gris.
- Para cada punto en `puntos_ojo_derecho + puntos_ojo_izquierdo + puntos_boca + [iris_derecho, iris_izquierdo]`: `cv2.circle` radio 2, color amarillo (mismo estilo que el loop de ojos que ya existe hoy en `main.py`).

No pasa por `dsd/db.py` ni afecta persistencia — es puro dibujo sobre el frame ya capturado, mismo tipo de efecto lateral que el `cv2.putText` del texto de sesión que ya existe en `main.py`.

### `dsd/main.py` — flag e integración

```python
def main(mostrar_malla: bool = False) -> None:
    ...
```

En el bloque `if estado.estado == Estado.ACTIVA:` ya existente, donde hoy está:

```python
for x, y in landmarks.puntos_ojo_derecho + landmarks.puntos_ojo_izquierdo:
    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)
```

pasa a ser:

```python
if mostrar_malla:
    dibujar_malla_debug(frame, landmarks)
else:
    for x, y in landmarks.puntos_ojo_derecho + landmarks.puntos_ojo_izquierdo:
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)
```

`if __name__ == "__main__":` gana parseo de argumentos (mismo patrón `argparse` que ya usa `dsd/enroll.py`):

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deteccion de somnolencia y distraccion del conductor.")
    parser.add_argument("--malla", action="store_true", help="Dibuja la malla facial completa y los puntos de control de deteccion, en vez de la superposicion normal.")
    args = parser.parse_args()
    main(mostrar_malla=args.malla)
```

## Testing

- **Automatizable:** `tests/test_debug_draw.py` — smoke test de que `CONEXIONES_TESELACION` (dato externo de Mediapipe del que dependemos) tiene todas las conexiones con índices `start`/`end` dentro de `[0, 477]` y no está vacía. Protege contra un cambio de versión de Mediapipe que altere esa tabla sin que nos demos cuenta. No requiere cámara.
- **Manual (cámara real):** `python -m dsd.main --malla` con una sesión activa — confirmar que la malla se dibuja sobre la cara siguiendo el movimiento, que los puntos de control amarillos coinciden con ojos/iris/boca, y que sin el flag (`python -m dsd.main`) el comportamiento es idéntico al actual (solo puntos de ojo).

## Fuera de alcance

Activar la detección de landmarks fuera de `Estado.ACTIVA`, indicador en pantalla de que el modo malla está activo, cualquier cambio a la lógica de somnolencia/distracción/reconocimiento, persistencia de las imágenes/capturas de la malla.
