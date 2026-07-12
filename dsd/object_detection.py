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
