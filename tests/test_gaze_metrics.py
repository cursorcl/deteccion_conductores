import pytest

from dsd.gaze_metrics import calcular_gaze_ratio

# Mismo orden de 6 puntos EAR: (esquina_externa, parpado_superior_1,
# parpado_superior_2, esquina_interna, parpado_inferior_2, parpado_inferior_1)
PUNTOS_OJO = [
    (0.0, 0.0), (0.3, -0.15), (0.7, -0.15), (1.0, 0.0), (0.7, 0.15), (0.3, 0.15)
]


def test_iris_centrado_da_ratios_0_5():
    ratio_h, ratio_v = calcular_gaze_ratio((0.5, 0.0), PUNTOS_OJO)
    assert ratio_h == pytest.approx(0.5)
    assert ratio_v == pytest.approx(0.5)


def test_iris_cerca_de_esquina_externa_da_ratio_horizontal_bajo():
    ratio_h, _ = calcular_gaze_ratio((0.05, 0.0), PUNTOS_OJO)
    assert ratio_h == pytest.approx(0.05)


def test_iris_cerca_de_esquina_interna_da_ratio_horizontal_alto():
    ratio_h, _ = calcular_gaze_ratio((0.95, 0.0), PUNTOS_OJO)
    assert ratio_h == pytest.approx(0.95)


def test_iris_cerca_del_parpado_superior_da_ratio_vertical_bajo():
    _, ratio_v = calcular_gaze_ratio((0.5, -0.14), PUNTOS_OJO)
    assert ratio_v == pytest.approx(0.01 / 0.3)


def test_iris_cerca_del_parpado_inferior_da_ratio_vertical_alto():
    _, ratio_v = calcular_gaze_ratio((0.5, 0.14), PUNTOS_OJO)
    assert ratio_v == pytest.approx(0.29 / 0.3)


def test_calcular_gaze_ratio_lanza_error_si_no_son_6_puntos():
    with pytest.raises(ValueError):
        calcular_gaze_ratio((0.5, 0.0), [(0.0, 0.0), (1.0, 1.0)])


def test_ancho_cero_retorna_ratio_horizontal_centrado():
    puntos_degenerados = [
        (0.5, 0.0), (0.5, -0.15), (0.5, -0.15), (0.5, 0.0), (0.5, 0.15), (0.5, 0.15)
    ]
    ratio_h, _ = calcular_gaze_ratio((0.5, 0.0), puntos_degenerados)
    assert ratio_h == 0.5
