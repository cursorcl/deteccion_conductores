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
