import pytest

from dsd.sustained_timer import EstadoTemporizadorSostenido, procesar_temporizador_sostenido

UMBRAL = 2.0
COOLDOWN = 30.0


def test_estado_inicial_no_tiene_temporizador_activo():
    estado = EstadoTemporizadorSostenido()
    assert estado.inicio is None
    assert estado.ultimo_disparo is None


def test_inactivo_no_acumula():
    estado = EstadoTemporizadorSostenido()
    nuevo_estado, valor = procesar_temporizador_sostenido(
        estado, condicion_activa=False, hubo_hueco=False, timestamp=0.0,
        umbral_segundos=UMBRAL, cooldown_segundos=COOLDOWN,
    )
    assert valor is None
    assert nuevo_estado.inicio is None


def test_activo_breve_no_dispara():
    estado = EstadoTemporizadorSostenido()
    for t in [0.0, 1.0]:
        estado, valor = procesar_temporizador_sostenido(
            estado, True, False, t, UMBRAL, COOLDOWN,
        )
        assert valor is None


def test_activo_exactamente_en_el_limite_no_dispara():
    estado = EstadoTemporizadorSostenido()
    for t in [0.0, 2.0]:
        estado, valor = procesar_temporizador_sostenido(
            estado, True, False, t, UMBRAL, COOLDOWN,
        )
    assert valor is None


def test_activo_sostenido_dispara():
    estado = EstadoTemporizadorSostenido()
    for t in [0.0, 1.0, 2.0]:
        estado, _ = procesar_temporizador_sostenido(estado, True, False, t, UMBRAL, COOLDOWN)
    estado, valor = procesar_temporizador_sostenido(estado, True, False, 2.1, UMBRAL, COOLDOWN)
    assert valor == pytest.approx(2.1)


def test_cooldown_bloquea_redisparo():
    estado = EstadoTemporizadorSostenido()
    for t in [0.0, 2.1]:
        estado, valor = procesar_temporizador_sostenido(estado, True, False, t, UMBRAL, COOLDOWN)
    assert valor == pytest.approx(2.1)
    estado, valor = procesar_temporizador_sostenido(estado, True, False, 20.0, UMBRAL, COOLDOWN)
    assert valor is None
    estado, valor = procesar_temporizador_sostenido(estado, True, False, 31.5, UMBRAL, COOLDOWN)
    assert valor is None


def test_redispara_tras_cooldown():
    estado = EstadoTemporizadorSostenido()
    for t in [0.0, 2.1, 20.0, 31.5]:
        estado, valor = procesar_temporizador_sostenido(estado, True, False, t, UMBRAL, COOLDOWN)
    # Ultimo disparo en t=2.1; cooldown de 30s expira en t=32.1.
    estado, valor = procesar_temporizador_sostenido(estado, True, False, 32.2, UMBRAL, COOLDOWN)
    assert valor == pytest.approx(32.2)


def test_apertura_reinicia_el_temporizador():
    estado = EstadoTemporizadorSostenido()
    estado, _ = procesar_temporizador_sostenido(estado, True, False, 0.0, UMBRAL, COOLDOWN)
    estado, _ = procesar_temporizador_sostenido(estado, False, False, 0.5, UMBRAL, COOLDOWN)
    assert estado.inicio is None
    estado, _ = procesar_temporizador_sostenido(estado, True, False, 0.6, UMBRAL, COOLDOWN)
    assert estado.inicio == 0.6


def test_hueco_reinicia_el_temporizador_aunque_condicion_siga_activa():
    # Reproduce el hallazgo de revision: si la condicion estaba activa antes
    # de un hueco prolongado (p.ej. el rostro no se detecto durante un
    # tramo largo) y sigue activa al reanudar, el temporizador NO debe
    # asumir que estuvo activo continuamente durante el hueco -- debe
    # reiniciar su "inicio" en el timestamp posterior al hueco.
    estado = EstadoTemporizadorSostenido()
    estado, _ = procesar_temporizador_sostenido(estado, True, False, 0.0, UMBRAL, COOLDOWN)
    estado, valor = procesar_temporizador_sostenido(estado, True, True, 50.0, UMBRAL, COOLDOWN)
    assert estado.inicio == 50.0
    assert valor is None


def test_sin_hueco_mantiene_el_inicio_original():
    estado = EstadoTemporizadorSostenido()
    estado, _ = procesar_temporizador_sostenido(estado, True, False, 0.0, UMBRAL, COOLDOWN)
    estado, _ = procesar_temporizador_sostenido(estado, True, False, 1.0, UMBRAL, COOLDOWN)
    assert estado.inicio == 0.0
