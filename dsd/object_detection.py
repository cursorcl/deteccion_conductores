from dataclasses import dataclass
from typing import List, Optional

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

RUTA_MODELO = "models/efficientdet_lite0.tflite"

# El detector se crea de forma perezosa (lazy) la primera vez que se use en
# detectar_objetos(), no al importar el modulo. Esto permite que codigo que
# importa solo ObjetoDetectado (ej. tests) no requiera el archivo del modelo
# en disco. El umbral de confianza aplicado aqui (score_threshold) es solo
# un piso bajo a nivel del modelo para no acumular ruido de detecciones muy
# debiles -- la decision real de "esto cuenta como celular" vive en
# dsd/phone_state.py (config.confianza_umbral), igual que ear_umbral vive en
# drowsiness_state.py y no en face_mesh.py.
_detector: Optional[vision.ObjectDetector] = None


def _get_detector() -> vision.ObjectDetector:
    """Crea el ObjectDetector de forma perezosa si no existe, y lo cachea."""
    global _detector
    if _detector is None:
        base_options = BaseOptions(model_asset_path=RUTA_MODELO)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            max_results=10,
            score_threshold=0.2,
        )
        _detector = vision.ObjectDetector.create_from_options(options)
    return _detector


@dataclass
class ObjetoDetectado:
    etiqueta: str
    confianza: float


def detectar_objetos(frame) -> List[ObjetoDetectado]:
    try:
        detector = _get_detector()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        resultado = detector.detect(imagen_mp)
    except Exception:
        return []

    objetos = []
    for deteccion in resultado.detections:
        for categoria in deteccion.categories:
            objetos.append(
                ObjetoDetectado(etiqueta=categoria.category_name, confianza=categoria.score)
            )
    return objetos
