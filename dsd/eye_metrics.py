import math
from typing import Sequence, Tuple


def calcular_ear(puntos_ojo: Sequence[Tuple[float, float]]) -> float:
    """Calcula el Eye Aspect Ratio (EAR) segun Soukupova & Cech, 2016.

    `puntos_ojo` debe tener exactamente 6 puntos (x, y) en el orden estandar:
    (esquina_externa, parpado_superior_1, parpado_superior_2, esquina_interna,
    parpado_inferior_2, parpado_inferior_1).

    EAR = (dist(p2, p6) + dist(p3, p5)) / (2 * dist(p1, p4))

    El resultado es invariante a la escala (distancia de la camara al rostro):
    es una razon entre distancias, no una distancia absoluta.
    """
    if len(puntos_ojo) != 6:
        raise ValueError(
            "calcular_ear requiere exactamente 6 puntos (esquina_externa, "
            "parpado_superior_1, parpado_superior_2, esquina_interna, "
            f"parpado_inferior_2, parpado_inferior_1); se recibieron {len(puntos_ojo)}."
        )

    p1, p2, p3, p4, p5, p6 = puntos_ojo

    def distancia(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    vertical = distancia(p2, p6) + distancia(p3, p5)
    horizontal = distancia(p1, p4)

    if horizontal == 0.0:
        return 0.0

    return vertical / (2 * horizontal)
