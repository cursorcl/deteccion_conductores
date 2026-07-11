import pytest

from dsd.mouth_metrics import calcular_mar

# Puntos sintéticos en el orden estándar de 6 puntos MAR:
# (comisura_izquierda, labio_superior_izquierdo, labio_superior_derecho,
#  comisura_derecha, labio_inferior_derecho, labio_inferior_izquierdo)
BOCA_CERRADA = [
    (0.0, 0.0), (0.3, -0.05), (0.7, -0.05), (1.0, 0.0), (0.7, 0.05), (0.3, 0.05)
]
BOCA_ABIERTA = [
    (0.0, 0.0), (0.3, -0.4), (0.7, -0.4), (1.0, 0.0), (0.7, 0.4), (0.3, 0.4)
]


def test_boca_cerrada_da_mar_bajo():
    assert calcular_mar(BOCA_CERRADA) == pytest.approx(0.1)


def test_boca_abierta_da_mar_alto():
    assert calcular_mar(BOCA_ABIERTA) == pytest.approx(0.8)


def test_mar_es_invariante_a_la_escala():
    boca_escalada = [(x * 100.0, y * 100.0) for x, y in BOCA_ABIERTA]
    assert calcular_mar(boca_escalada) == pytest.approx(calcular_mar(BOCA_ABIERTA))


def test_boca_abierta_supera_umbral_tipico():
    assert calcular_mar(BOCA_ABIERTA) > 0.6


def test_boca_cerrada_no_supera_umbral_tipico():
    assert calcular_mar(BOCA_CERRADA) < 0.6


def test_calcular_mar_lanza_error_si_no_son_6_puntos():
    with pytest.raises(ValueError):
        calcular_mar([(0.0, 0.0), (1.0, 1.0)])
