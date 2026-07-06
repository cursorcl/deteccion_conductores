import math
from typing import Sequence, Tuple


def calcular_yaw_pitch(matriz_rotacion: Sequence[Sequence[float]]) -> Tuple[float, float]:
    """Extrae (yaw, pitch), en grados, desde una matriz de rotacion 3x3.

    Asume la convencion R = Ry(yaw) @ Rx(pitch) @ Rz(roll) (angulos de
    Euler intrinsecos): yaw es la rotacion horizontal (eje Y, girar la
    cabeza hacia un costado); pitch es la rotacion vertical (eje X,
    inclinar la cabeza hacia arriba/abajo). roll (inclinacion lateral,
    eje Z) no se calcula porque no se usa para detectar distraccion.

    Formula de extraccion (valida para |pitch| < 90 grados, rango muy
    por encima de lo que ocurre al conducir):
        pitch = asin(-R[1][2])
        yaw   = atan2(R[0][2], R[2][2])
    """
    r02 = matriz_rotacion[0][2]
    r12 = matriz_rotacion[1][2]
    r22 = matriz_rotacion[2][2]

    seno_pitch = max(-1.0, min(1.0, -r12))
    pitch_rad = math.asin(seno_pitch)
    yaw_rad = math.atan2(r02, r22)

    return math.degrees(yaw_rad), math.degrees(pitch_rad)
