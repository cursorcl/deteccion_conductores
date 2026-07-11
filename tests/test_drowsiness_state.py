from dsd.config import ConfigSomnolencia
from dsd.drowsiness_state import (
    EstadoSomnolencia,
    EventoSomnolencia,
    Muestra,
    estado_inicial_somnolencia,
    procesar_somnolencia,
)

CONFIG = ConfigSomnolencia(
    ear_umbral=0.21,
    microsueno_segundos=1.5,
    perclos_ventana_segundos=60.0,
    perclos_umbral=0.15,
    cooldown_segundos=30.0,
    perclos_cobertura_minima=0.5,
    gap_maximo_segundos=1.0,
    mar_umbral=0.6,
    bostezo_min_segundos=1.5,
    bostezo_ventana_segundos=300.0,
    bostezo_umbral_cantidad=3.0,
)

EAR_CERRADO = 0.10
EAR_ABIERTO = 0.30
MAR_CERRADO = 0.10
MAR_ABIERTO = 0.80


def test_ojo_abierto_no_acumula_cierre():
    estado = estado_inicial_somnolencia()
    nuevo_estado, eventos = procesar_somnolencia(
        estado, EAR_ABIERTO, MAR_CERRADO, timestamp=0.0, config=CONFIG
    )
    assert eventos == []
    assert nuevo_estado.cierre_inicio is None


def test_cierre_breve_no_dispara_microsueno():
    estado = estado_inicial_somnolencia()
    for t in [0.0, 0.5, 1.0]:
        estado, eventos = procesar_somnolencia(
            estado, EAR_CERRADO, MAR_CERRADO, timestamp=t, config=CONFIG
        )
        assert eventos == []


def test_cierre_exactamente_en_el_limite_no_dispara():
    # Pasos densos (<= gap_maximo_segundos) para que el chequeo de hueco no
    # interfiera con la condicion de limite que este test quiere ejercitar.
    estado = estado_inicial_somnolencia()
    for t in [0.0, 0.5, 1.0, 1.5]:
        estado, eventos = procesar_somnolencia(
            estado, EAR_CERRADO, MAR_CERRADO, timestamp=t, config=CONFIG
        )
    assert eventos == []


def test_cierre_sostenido_dispara_microsueno():
    estado = estado_inicial_somnolencia()
    for t in [0.0, 0.5, 1.0, 1.5]:
        estado, eventos = procesar_somnolencia(
            estado, EAR_CERRADO, MAR_CERRADO, timestamp=t, config=CONFIG
        )
    estado, eventos = procesar_somnolencia(
        estado, EAR_CERRADO, MAR_CERRADO, timestamp=1.6, config=CONFIG
    )
    assert eventos == [EventoSomnolencia(tipo="microsueno", valor=1.6)]


def test_microsueno_no_re_dispara_dentro_del_cooldown():
    # Muestreo continuo (paso 0.5s, bajo gap_maximo_segundos) para simular
    # ojos cerrados de forma ininterrumpida durante todo el cooldown.
    estado = estado_inicial_somnolencia()
    eventos_microsueno = []
    t = 0.0
    while t <= 31.5:
        estado, eventos = procesar_somnolencia(
            estado, EAR_CERRADO, MAR_CERRADO, timestamp=t, config=CONFIG
        )
        eventos_microsueno += [e for e in eventos if e.tipo == "microsueno"]
        t += 0.5
    # Un unico disparo (en t=2.0); el cooldown de 30s sigue activo durante
    # el resto del tramo (expira recien en t=32.0).
    assert len(eventos_microsueno) == 1
    assert eventos_microsueno[0].valor == 2.0


def test_microsueno_re_dispara_tras_cooldown_si_sigue_cerrado():
    estado = estado_inicial_somnolencia()
    eventos_microsueno = []
    t = 0.0
    while t <= 32.5:
        estado, eventos = procesar_somnolencia(
            estado, EAR_CERRADO, MAR_CERRADO, timestamp=t, config=CONFIG
        )
        eventos_microsueno += [e for e in eventos if e.tipo == "microsueno"]
        t += 0.5
    assert len(eventos_microsueno) == 2
    assert eventos_microsueno[0].valor == 2.0
    assert eventos_microsueno[1].valor == 32.0


def test_apertura_de_ojos_reinicia_temporizador_microsueno():
    estado = estado_inicial_somnolencia()
    estado, _ = procesar_somnolencia(estado, EAR_CERRADO, MAR_CERRADO, timestamp=0.0, config=CONFIG)
    estado, _ = procesar_somnolencia(estado, EAR_ABIERTO, MAR_CERRADO, timestamp=0.5, config=CONFIG)
    assert estado.cierre_inicio is None
    estado, _ = procesar_somnolencia(estado, EAR_CERRADO, MAR_CERRADO, timestamp=0.6, config=CONFIG)
    assert estado.cierre_inicio == 0.6


def test_microsueno_no_dispara_con_valor_inflado_tras_hueco_prolongado():
    # Reproduce el hallazgo de la revision final del detector de
    # distraccion (aplicable tambien aqui): si el rostro no se detecta
    # durante un tramo largo (frames descartados por completo, sin llamar
    # a procesar_somnolencia) y luego se retoma con los ojos ya cerrados,
    # el temporizador de microsueno NO debe asumir que estuvo cerrado
    # desde antes del hueco.
    estado = estado_inicial_somnolencia()
    estado, _ = procesar_somnolencia(estado, EAR_ABIERTO, MAR_CERRADO, timestamp=0.0, config=CONFIG)
    # Hueco prolongado: sin llamadas entre t=0 y t=50 (rostro no detectado).
    estado, eventos = procesar_somnolencia(estado, EAR_CERRADO, MAR_CERRADO, timestamp=50.0, config=CONFIG)
    assert eventos == []
    assert estado.cierre_inicio == 50.0


def test_perclos_no_evalua_antes_de_completar_ventana():
    estado = estado_inicial_somnolencia()
    eventos_perclos = []
    t = 0.0
    while t < 60.0:
        estado, eventos = procesar_somnolencia(
            estado, EAR_CERRADO, MAR_CERRADO, timestamp=t, config=CONFIG
        )
        eventos_perclos += [e for e in eventos if e.tipo == "perclos"]
        t += 1.0
    assert eventos_perclos == []


def test_perclos_dispara_cuando_fraccion_cerrada_supera_umbral():
    estado = estado_inicial_somnolencia()
    eventos_perclos = []
    t = 0.0
    while t <= 65.0:
        estado, eventos = procesar_somnolencia(
            estado, EAR_CERRADO, MAR_CERRADO, timestamp=t, config=CONFIG
        )
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
        estado, eventos = procesar_somnolencia(estado, ear, MAR_CERRADO, timestamp=t, config=CONFIG)
        eventos_perclos += [e for e in eventos if e.tipo == "perclos"]
        t += 1.0
    assert eventos_perclos == []


def test_muestras_antiguas_se_recortan_fuera_de_la_ventana():
    estado = estado_inicial_somnolencia()
    t = 0.0
    while t <= 65.0:
        estado, _ = procesar_somnolencia(estado, EAR_CERRADO, MAR_CERRADO, timestamp=t, config=CONFIG)
        t += 1.0
    assert all(
        m.timestamp >= t - 1.0 - CONFIG.perclos_ventana_segundos for m in estado.muestras
    )


def test_estado_inicial_no_tiene_muestras():
    estado = estado_inicial_somnolencia()
    assert estado.muestras == []
    assert estado.cierre_inicio is None
    assert estado.bostezos == []
    assert estado.boca_abierta_inicio is None


def test_microsueno_y_perclos_pueden_dispararse_en_el_mismo_llamado():
    # Construye el estado previo directamente (en vez de hacerlo evolucionar
    # con muchas llamadas) para forzar que ambos temporizadores esten listos
    # para disparar en la misma llamada. Bajo cierre continuo, microsueno
    # (periodo 30s desde su primer disparo en t=2) y perclos (periodo 30s
    # desde su primer disparo en t=60) tienen fases distintas y nunca
    # coinciden solos -- por eso se fuerza la precondicion explicitamente.
    muestras_previas = [Muestra(timestamp=float(t), cerrado=True) for t in range(0, 60)]
    estado = EstadoSomnolencia(
        muestras=muestras_previas,
        cierre_inicio=58.0,
        ultimo_disparo_microsueno=None,
        ultimo_disparo_perclos=None,
        primer_timestamp=0.0,
        ultimo_procesado=59.0,
    )
    estado, eventos = procesar_somnolencia(estado, EAR_CERRADO, MAR_CERRADO, timestamp=60.0, config=CONFIG)
    tipos = {e.tipo for e in eventos}
    assert tipos == {"microsueno", "perclos"}


def test_boca_cerrada_no_acumula_apertura():
    estado = estado_inicial_somnolencia()
    nuevo_estado, eventos = procesar_somnolencia(
        estado, EAR_ABIERTO, MAR_CERRADO, timestamp=0.0, config=CONFIG
    )
    assert eventos == []
    assert nuevo_estado.boca_abierta_inicio is None


def test_apertura_breve_no_dispara_bostezo():
    estado = estado_inicial_somnolencia()
    for t in [0.0, 0.5, 1.0]:
        estado, eventos = procesar_somnolencia(
            estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=t, config=CONFIG
        )
        assert eventos == []


def test_apertura_sostenida_dispara_bostezo():
    estado = estado_inicial_somnolencia()
    for t in [0.0, 0.5, 1.0, 1.5]:
        estado, eventos = procesar_somnolencia(
            estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=t, config=CONFIG
        )
    estado, eventos = procesar_somnolencia(
        estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=1.6, config=CONFIG
    )
    assert eventos == [EventoSomnolencia(tipo="bostezo", valor=1.6)]


def test_bostezo_no_re_dispara_dentro_del_cooldown():
    estado = estado_inicial_somnolencia()
    eventos_bostezo = []
    t = 0.0
    while t <= 31.5:
        estado, eventos = procesar_somnolencia(
            estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=t, config=CONFIG
        )
        eventos_bostezo += [e for e in eventos if e.tipo == "bostezo"]
        t += 0.5
    assert len(eventos_bostezo) == 1
    assert eventos_bostezo[0].valor == 2.0


def test_cierre_de_boca_reinicia_temporizador_bostezo():
    estado = estado_inicial_somnolencia()
    estado, _ = procesar_somnolencia(estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=0.0, config=CONFIG)
    estado, _ = procesar_somnolencia(estado, EAR_ABIERTO, MAR_CERRADO, timestamp=0.5, config=CONFIG)
    assert estado.boca_abierta_inicio is None
    estado, _ = procesar_somnolencia(estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=0.6, config=CONFIG)
    assert estado.boca_abierta_inicio == 0.6


def test_bostezo_no_dispara_con_valor_inflado_tras_hueco_prolongado():
    estado = estado_inicial_somnolencia()
    estado, _ = procesar_somnolencia(estado, EAR_ABIERTO, MAR_CERRADO, timestamp=0.0, config=CONFIG)
    estado, eventos = procesar_somnolencia(estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=50.0, config=CONFIG)
    assert eventos == []
    assert estado.boca_abierta_inicio == 50.0


def test_fatiga_bostezos_no_dispara_antes_de_alcanzar_la_cantidad():
    estado = estado_inicial_somnolencia()
    eventos_fatiga = []
    for inicio in [0.0, 40.0]:
        for t in [inicio, inicio + 0.5, inicio + 1.0, inicio + 1.5, inicio + 1.6]:
            estado, eventos = procesar_somnolencia(
                estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=t, config=CONFIG
            )
            eventos_fatiga += [e for e in eventos if e.tipo == "fatiga_bostezos"]
    assert eventos_fatiga == []
    assert len(estado.bostezos) == 2


def test_fatiga_bostezos_dispara_al_alcanzar_la_cantidad_en_la_ventana():
    estado = estado_inicial_somnolencia()
    eventos_fatiga = []
    for inicio in [0.0, 40.0, 80.0]:
        for t in [inicio, inicio + 0.5, inicio + 1.0, inicio + 1.5, inicio + 1.6]:
            estado, eventos = procesar_somnolencia(
                estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=t, config=CONFIG
            )
            eventos_fatiga += [e for e in eventos if e.tipo == "fatiga_bostezos"]
    assert len(eventos_fatiga) == 1
    assert eventos_fatiga[0].valor == 3.0


def test_bostezos_fuera_de_la_ventana_se_recortan():
    estado = estado_inicial_somnolencia()
    for inicio in [0.0, 40.0, 80.0]:
        for t in [inicio, inicio + 0.5, inicio + 1.0, inicio + 1.5, inicio + 1.6]:
            estado, _ = procesar_somnolencia(
                estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=t, config=CONFIG
            )
    # Cuarto bostezo bien fuera de la ventana de 300s respecto al primero:
    # solo deberian quedar los bostezos dentro de los ultimos 300s.
    inicio = 350.0
    for t in [inicio, inicio + 0.5, inicio + 1.0, inicio + 1.5, inicio + 1.6]:
        estado, _ = procesar_somnolencia(
            estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=t, config=CONFIG
        )
    assert all(
        ts >= (inicio + 1.6) - CONFIG.bostezo_ventana_segundos for ts in estado.bostezos
    )


def test_fatiga_bostezos_no_re_dispara_dentro_del_cooldown():
    estado = estado_inicial_somnolencia()
    eventos_fatiga = []
    for inicio in [0.0, 40.0, 80.0]:
        for t in [inicio, inicio + 0.5, inicio + 1.0, inicio + 1.5, inicio + 1.6]:
            estado, eventos = procesar_somnolencia(
                estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=t, config=CONFIG
            )
            eventos_fatiga += [e for e in eventos if e.tipo == "fatiga_bostezos"]
    # Tercer bostezo (t=81.6) hace que la cuenta llegue a 3 y dispara
    # fatiga_bostezos. Los saltos de tiempo siguientes (>gap_maximo_segundos)
    # representan tramos con la boca cerrada -- no deberian generar nuevos
    # bostezos, y mientras la cuenta en ventana siga en 3, fatiga_bostezos
    # no debe volver a dispararse antes de que expire su propio cooldown.
    for t in [90.0, 100.0, 109.9]:
        estado, eventos = procesar_somnolencia(
            estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=t, config=CONFIG
        )
        eventos_fatiga += [e for e in eventos if e.tipo == "fatiga_bostezos"]
    assert len(eventos_fatiga) == 1


def test_fatiga_bostezos_re_dispara_tras_cooldown_si_la_cuenta_sigue_alta():
    estado = estado_inicial_somnolencia()
    eventos_fatiga = []
    for inicio in [0.0, 40.0, 80.0]:
        for t in [inicio, inicio + 0.5, inicio + 1.0, inicio + 1.5, inicio + 1.6]:
            estado, eventos = procesar_somnolencia(
                estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=t, config=CONFIG
            )
            eventos_fatiga += [e for e in eventos if e.tipo == "fatiga_bostezos"]
    # Cooldown de fatiga_bostezos expira en t=81.6+30=111.6; a t=112.0 el
    # conteo de bostezos en ventana sigue en 3, asi que vuelve a disparar.
    estado, eventos = procesar_somnolencia(
        estado, EAR_ABIERTO, MAR_ABIERTO, timestamp=112.0, config=CONFIG
    )
    eventos_fatiga += [e for e in eventos if e.tipo == "fatiga_bostezos"]
    assert len(eventos_fatiga) == 2
