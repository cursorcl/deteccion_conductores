import pytest

from dsd.eye_metrics import calcular_ear

# Puntos sintéticos en el orden estándar de 6 puntos EAR:
# (esquina_externa, parpado_superior_1, parpado_superior_2, esquina_interna,
#  parpado_inferior_2, parpado_inferior_1)
OJO_ABIERTO = [
    (0.0, 0.0), (0.3, -0.15), (0.7, -0.15), (1.0, 0.0), (0.7, 0.15), (0.3, 0.15)
]
OJO_CERRADO = [
    (0.0, 0.0), (0.3, -0.025), (0.7, -0.025), (1.0, 0.0), (0.7, 0.025), (0.3, 0.025)
]


def test_ojo_abierto_da_ear_alto():
    assert calcular_ear(OJO_ABIERTO) == pytest.approx(0.3)


def test_ojo_cerrado_da_ear_bajo():
    assert calcular_ear(OJO_CERRADO) == pytest.approx(0.05)


def test_ear_es_invariante_a_la_escala():
    ojo_escalado = [(x * 100.0, y * 100.0) for x, y in OJO_ABIERTO]
    assert calcular_ear(ojo_escalado) == pytest.approx(calcular_ear(OJO_ABIERTO))


def test_ojo_abierto_supera_umbral_tipico():
    assert calcular_ear(OJO_ABIERTO) > 0.21


def test_ojo_cerrado_no_supera_umbral_tipico():
    assert calcular_ear(OJO_CERRADO) < 0.21


def test_calcular_ear_lanza_error_si_no_son_6_puntos():
    with pytest.raises(ValueError):
        calcular_ear([(0.0, 0.0), (1.0, 1.0)])
