import os
from typing import Optional, Tuple

from deepface import DeepFace

DIRECTORIO_CONDUCTORES = "known_drivers"
UMBRAL_ESTRICTO = 0.68
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "mtcnn"


def reconocer_conductor(frame) -> Optional[Tuple[str, float]]:
    try:
        # enforce_detection=True: con False, un frame sin rostro real (camara
        # tapada, encuadre vacio) generaba una deteccion falsa de todo el frame
        # cuyo embedding a veces caia dentro del umbral, dando un match falso.
        resultados = DeepFace.find(
            img_path=frame,
            db_path=DIRECTORIO_CONDUCTORES,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            align=True,
            silent=True,
            threshold=2.0,
        )
    except Exception:
        return None

    for df in resultados:
        if df.empty:
            continue
        distancia = df["distance"][0]
        if distancia <= UMBRAL_ESTRICTO:
            ruta_identidad = df["identity"][0]
            nombre = ruta_identidad.split(os.path.sep)[-2]
            return nombre, distancia

    return None
