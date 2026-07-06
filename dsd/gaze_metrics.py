from typing import Sequence, Tuple


def calcular_gaze_ratio(
    iris: Tuple[float, float], puntos_ojo: Sequence[Tuple[float, float]]
) -> Tuple[float, float]:
    """Calcula la posicion relativa del iris dentro del contorno del ojo.

    `puntos_ojo` debe tener el mismo orden de 6 puntos usado para EAR:
    (esquina_externa, parpado_superior_1, parpado_superior_2,
    esquina_interna, parpado_inferior_2, parpado_inferior_1).

    Retorna (ratio_horizontal, ratio_vertical): 0.5 en cualquiera de los
    dos significa iris centrado en esa dimension; valores que se alejan
    de 0.5 indican mirada desviada hacia una esquina/parpado. El eje
    horizontal va de esquina_externa (0.0) a esquina_interna (1.0); el
    eje vertical va de parpado_superior (0.0) a parpado_inferior (1.0).
    """
    if len(puntos_ojo) != 6:
        raise ValueError(
            "calcular_gaze_ratio requiere exactamente 6 puntos de ojo "
            f"(mismo orden que EAR); se recibieron {len(puntos_ojo)}."
        )

    esquina_externa, sup1, sup2, esquina_interna, inf2, inf1 = puntos_ojo
    iris_x, iris_y = iris

    ancho = esquina_interna[0] - esquina_externa[0]
    ratio_horizontal = (iris_x - esquina_externa[0]) / ancho if ancho != 0.0 else 0.5

    y_superior = (sup1[1] + sup2[1]) / 2
    y_inferior = (inf1[1] + inf2[1]) / 2
    alto = y_inferior - y_superior
    ratio_vertical = (iris_y - y_superior) / alto if alto != 0.0 else 0.5

    return ratio_horizontal, ratio_vertical
