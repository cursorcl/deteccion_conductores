from dsd.config import ConfigSomnolencia
from dsd.drowsiness_state import (
    EventoSomnolencia,
    estado_inicial_somnolencia,
    procesar_ear,
)

CONFIG = ConfigSomnolencia(
    ear_umbral=0.21,
    microsueno_segundos=1.5,
    perclos_ventana_segundos=60.0,
    perclos_umbral=0.15,
    cooldown_segundos=30.0,
)

EAR_CERRADO = 0.10
EAR_ABIERTO = 0.30


def test_ojo_abierto_no_acumula_cierre():
    estado = estado_inicial_somnolencia()
    nuevo_estado, eventos = procesar_ear(estado, EAR_ABIERTO, timestamp=0.0, config=CONFIG)
    assert eventos == []
    assert nuevo_estado.cierre_inicio is None


def test_cierre_breve_no_dispara_microsueno():
    estado = estado_inicial_somnolencia()
    for t in [0.0, 0.5, 1.0]:
        estado, eventos = procesar_ear(estado, EAR_CERRADO, timestamp=t, config=CONFIG)
        assert eventos == []


def test_cierre_exactamente_en_el_limite_no_dispara():
    estado = estado_inicial_somnolencia()
    for t in [0.0, 1.5]:
        estado, eventos = procesar_ear(estado, EAR_CERRADO, timestamp=t, config=CONFIG)
    assert eventos == []


def test_cierre_sostenido_dispara_microsueno():
    estado = estado_inicial_somnolencia()
    for t in [0.0, 0.5, 1.0, 1.5]:
        estado, eventos = procesar_ear(estado, EAR_CERRADO, timestamp=t, config=CONFIG)
    estado, eventos = procesar_ear(estado, EAR_CERRADO, timestamp=1.6, config=CONFIG)
    assert eventos == [EventoSomnolencia(tipo="microsueno", valor=1.6)]


def test_microsueno_no_re_dispara_dentro_del_cooldown():
    estado = estado_inicial_somnolencia()
    for t in [0.0, 1.6]:
        estado, eventos = procesar_ear(estado, EAR_CERRADO, timestamp=t, config=CONFIG)
    estado, eventos = procesar_ear(estado, EAR_CERRADO, timestamp=20.0, config=CONFIG)
    assert eventos == []
    estado, eventos = procesar_ear(estado, EAR_CERRADO, timestamp=31.5, config=CONFIG)
    assert eventos == []


def test_microsueno_re_dispara_tras_cooldown_si_sigue_cerrado():
    estado = estado_inicial_somnolencia()
    for t in [0.0, 1.6, 20.0, 31.5]:
        estado, eventos = procesar_ear(estado, EAR_CERRADO, timestamp=t, config=CONFIG)
    estado, eventos = procesar_ear(estado, EAR_CERRADO, timestamp=31.7, config=CONFIG)
    assert eventos == [EventoSomnolencia(tipo="microsueno", valor=31.7)]


def test_apertura_de_ojos_reinicia_temporizador_microsueno():
    estado = estado_inicial_somnolencia()
    estado, _ = procesar_ear(estado, EAR_CERRADO, timestamp=0.0, config=CONFIG)
    estado, _ = procesar_ear(estado, EAR_ABIERTO, timestamp=0.5, config=CONFIG)
    assert estado.cierre_inicio is None
    estado, _ = procesar_ear(estado, EAR_CERRADO, timestamp=0.6, config=CONFIG)
    assert estado.cierre_inicio == 0.6


def test_perclos_no_evalua_antes_de_completar_ventana():
    estado = estado_inicial_somnolencia()
    eventos_perclos = []
    t = 0.0
    while t < 60.0:
        estado, eventos = procesar_ear(estado, EAR_CERRADO, timestamp=t, config=CONFIG)
        eventos_perclos += [e for e in eventos if e.tipo == "perclos"]
        t += 1.0
    assert eventos_perclos == []


def test_perclos_dispara_cuando_fraccion_cerrada_supera_umbral():
    estado = estado_inicial_somnolencia()
    eventos_perclos = []
    t = 0.0
    while t <= 65.0:
        estado, eventos = procesar_ear(estado, EAR_CERRADO, timestamp=t, config=CONFIG)
        eventos_perclos += [e for e in eventos if e.tipo == "perclos"]
        t += 1.0
    assert len(eventos_perclos) >= 1
    assert eventos_perclos[0].valor == 1.0


def test_perclos_no_dispara_si_fraccion_bajo_umbral():
    estado = estado_inicial_somnolencia()
    eventos_perclos = []
    t = 0.0
    while t <= 65.0:
        # 1 de cada 10 muestras cerrada = 10% < 15% del umbral.
        cerrado = (int(t) % 10 == 0)
        ear = EAR_CERRADO if cerrado else EAR_ABIERTO
        estado, eventos = procesar_ear(estado, ear, timestamp=t, config=CONFIG)
        eventos_perclos += [e for e in eventos if e.tipo == "perclos"]
        t += 1.0
    assert eventos_perclos == []


def test_muestras_antiguas_se_recortan_fuera_de_la_ventana():
    estado = estado_inicial_somnolencia()
    t = 0.0
    while t <= 65.0:
        estado, _ = procesar_ear(estado, EAR_CERRADO, timestamp=t, config=CONFIG)
        t += 1.0
    assert all(
        m.timestamp >= t - 1.0 - CONFIG.perclos_ventana_segundos for m in estado.muestras
    )


def test_estado_inicial_no_tiene_muestras():
    estado = estado_inicial_somnolencia()
    assert estado.muestras == []
    assert estado.cierre_inicio is None


def test_microsueno_y_perclos_pueden_dispararse_en_el_mismo_llamado():
    estado = estado_inicial_somnolencia()
    t = 0.0
    while t < 60.0:
        estado, _ = procesar_ear(estado, EAR_CERRADO, timestamp=t, config=CONFIG)
        t += 1.0
    # El microsueno anterior disparo en t=32.0 (primer disparo en t=2.0,
    # cooldown de 30s activo hasta t=32.0); su propio cooldown de 30s no
    # expira hasta t=62.0, asi que el llamado final debe ser posterior a
    # ese punto para que ambos tipos de evento puedan dispararse juntos.
    estado, eventos = procesar_ear(estado, EAR_CERRADO, timestamp=62.1, config=CONFIG)
    tipos = {e.tipo for e in eventos}
    assert tipos == {"microsueno", "perclos"}
