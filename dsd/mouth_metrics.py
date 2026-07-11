import math
from typing import Sequence, Tuple


def calcular_mar(puntos_boca: Sequence[Tuple[float, float]]) -> float:
    """Calcula el Mouth Aspect Ratio (MAR), geometricamente identico al
    EAR (ver dsd/eye_metrics.py) pero aplicado a los 6 puntos de boca en
    el orden estandar: (comisura_izquierda, labio_superior_izquierdo,
    labio_superior_derecho, comisura_derecha, labio_inferior_derecho,
    labio_inferior_izquierdo).

    MAR = (dist(p2, p6) + dist(p3, p5)) / (2 * dist(p1, p4))

    A diferencia del EAR (donde el ojo cerrado da el valor bajo), aqui la
    boca cerrada da MAR bajo y la boca abierta (bostezo) da MAR alto.
    """
    if len(puntos_boca) != 6:
        raise ValueError(
            "calcular_mar requiere exactamente 6 puntos (comisura_izquierda, "
            "labio_superior_izquierdo, labio_superior_derecho, comisura_derecha, "
            f"labio_inferior_derecho, labio_inferior_izquierdo); se recibieron {len(puntos_boca)}."
        )

    p1, p2, p3, p4, p5, p6 = puntos_boca

    def distancia(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    vertical = distancia(p2, p6) + distancia(p3, p5)
    horizontal = distancia(p1, p4)

    if horizontal == 0.0:
        return 0.0

    return vertical / (2 * horizontal)
