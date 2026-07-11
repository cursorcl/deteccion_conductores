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
