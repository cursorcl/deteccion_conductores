import math

import pytest

from dsd.head_pose import calcular_yaw_pitch


def _matriz_rotacion(yaw_grados: float, pitch_grados: float, roll_grados: float = 0.0):
    """Construye una matriz de rotacion 3x3 sintetica R = Ry(yaw) @ Rx(pitch)
    @ Rz(roll), usada solo para generar datos de prueba (no depende de
    Mediapipe ni de camara real)."""
    yaw = math.radians(yaw_grados)
    pitch = math.radians(pitch_grados)
    roll = math.radians(roll_grados)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    return [
        [cy * cr + sy * sp * sr, -cy * sr + sy * sp * cr, sy * cp],
        [cp * sr, cp * cr, -sp],
        [-sy * cr + cy * sp * sr, sy * sr + cy * sp * cr, cy * cp],
    ]


def test_matriz_identidad_da_yaw_pitch_cero():
    yaw, pitch = calcular_yaw_pitch(_matriz_rotacion(0.0, 0.0))
    assert yaw == pytest.approx(0.0, abs=1e-9)
    assert pitch == pytest.approx(0.0, abs=1e-9)


def test_yaw_positivo_se_recupera_correctamente():
    yaw, pitch = calcular_yaw_pitch(_matriz_rotacion(30.0, 0.0))
    assert yaw == pytest.approx(30.0)
    assert pitch == pytest.approx(0.0, abs=1e-9)


def test_yaw_negativo_se_recupera_correctamente():
    yaw, pitch = calcular_yaw_pitch(_matriz_rotacion(-25.0, 0.0))
    assert yaw == pytest.approx(-25.0)
    assert pitch == pytest.approx(0.0, abs=1e-9)


def test_pitch_positivo_se_recupera_correctamente():
    yaw, pitch = calcular_yaw_pitch(_matriz_rotacion(0.0, 20.0))
    assert yaw == pytest.approx(0.0, abs=1e-9)
    assert pitch == pytest.approx(20.0)


def test_pitch_negativo_se_recupera_correctamente():
    yaw, pitch = calcular_yaw_pitch(_matriz_rotacion(0.0, -15.0))
    assert yaw == pytest.approx(0.0, abs=1e-9)
    assert pitch == pytest.approx(-15.0)


def test_yaw_y_pitch_combinados_se_recuperan_correctamente():
    yaw, pitch = calcular_yaw_pitch(_matriz_rotacion(15.0, 10.0))
    assert yaw == pytest.approx(15.0)
    assert pitch == pytest.approx(10.0)


def test_roll_no_afecta_yaw_ni_pitch_calculados():
    yaw, pitch = calcular_yaw_pitch(_matriz_rotacion(15.0, 10.0, roll_grados=45.0))
    assert yaw == pytest.approx(15.0)
    assert pitch == pytest.approx(10.0)
