from dsd.config import ConfigCelular
from dsd.object_detection import ObjetoDetectado
from dsd.phone_state import EventoCelular, estado_inicial_celular, procesar_objetos

CONFIG = ConfigCelular(
    confianza_umbral=0.5,
    celular_segundos=2.0,
    cooldown_segundos=30.0,
    gap_maximo_segundos=1.0,
)

CELULAR_CONFIANZA_ALTA = [ObjetoDetectado(etiqueta="cell phone", confianza=0.8)]
CELULAR_CONFIANZA_BAJA = [ObjetoDetectado(etiqueta="cell phone", confianza=0.3)]
OTRO_OBJETO = [ObjetoDetectado(etiqueta="person", confianza=0.9)]
SIN_OBJETOS = []


def test_estado_inicial_no_tiene_temporizador_activo():
    estado = estado_inicial_celular()
    assert estado.deteccion_inicio is None
    assert estado.ultimo_disparo is None
    assert estado.ultimo_procesado is None


def test_sin_objetos_no_acumula_deteccion():
    estado = estado_inicial_celular()
    nuevo_estado, eventos = procesar_objetos(estado, SIN_OBJETOS, timestamp=0.0, config=CONFIG)
    assert eventos == []
    assert nuevo_estado.deteccion_inicio is None


def test_otro_objeto_no_cuenta_como_celular():
    estado = estado_inicial_celular()
    nuevo_estado, eventos = procesar_objetos(estado, OTRO_OBJETO, timestamp=0.0, config=CONFIG)
    assert eventos == []
    assert nuevo_estado.deteccion_inicio is None


def test_confianza_baja_no_cuenta_como_celular_detectado():
    estado = estado_inicial_celular()
    nuevo_estado, eventos = procesar_objetos(
        estado, CELULAR_CONFIANZA_BAJA, timestamp=0.0, config=CONFIG
    )
    assert eventos == []
    assert nuevo_estado.deteccion_inicio is None


def test_deteccion_breve_no_dispara_uso_celular():
    estado = estado_inicial_celular()
    for t in [0.0, 1.0]:
        estado, eventos = procesar_objetos(
            estado, CELULAR_CONFIANZA_ALTA, timestamp=t, config=CONFIG
        )
        assert eventos == []


def test_deteccion_exactamente_en_el_limite_no_dispara():
    # Pasos densos (<= gap_maximo_segundos) para que el chequeo de hueco no
    # interfiera con la condicion de limite que este test quiere ejercitar.
    estado = estado_inicial_celular()
    for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
        estado, eventos = procesar_objetos(
            estado, CELULAR_CONFIANZA_ALTA, timestamp=t, config=CONFIG
        )
    assert eventos == []


def test_deteccion_sostenida_dispara_uso_celular():
    estado = estado_inicial_celular()
    for t in [0.0, 1.0, 2.0]:
        estado, _ = procesar_objetos(
            estado, CELULAR_CONFIANZA_ALTA, timestamp=t, config=CONFIG
        )
    estado, eventos = procesar_objetos(
        estado, CELULAR_CONFIANZA_ALTA, timestamp=2.1, config=CONFIG
    )
    assert eventos == [EventoCelular(tipo="uso_celular", valor=2.1)]


def test_dejar_de_detectar_reinicia_temporizador():
    estado = estado_inicial_celular()
    estado, _ = procesar_objetos(estado, CELULAR_CONFIANZA_ALTA, timestamp=0.0, config=CONFIG)
    estado, _ = procesar_objetos(estado, SIN_OBJETOS, timestamp=0.5, config=CONFIG)
    assert estado.deteccion_inicio is None
    estado, _ = procesar_objetos(estado, CELULAR_CONFIANZA_ALTA, timestamp=0.6, config=CONFIG)
    assert estado.deteccion_inicio == 0.6


def test_uso_celular_no_re_dispara_dentro_del_cooldown():
    # Muestreo continuo (paso 0.5s, bajo gap_maximo_segundos) para simular
    # celular detectado de forma ininterrumpida durante todo el cooldown.
    estado = estado_inicial_celular()
    eventos_celular = []
    t = 0.0
    while t <= 31.5:
        estado, eventos = procesar_objetos(
            estado, CELULAR_CONFIANZA_ALTA, timestamp=t, config=CONFIG
        )
        eventos_celular += [e for e in eventos if e.tipo == "uso_celular"]
        t += 0.5
    assert len(eventos_celular) == 1
    assert eventos_celular[0].valor == 2.5


def test_uso_celular_re_dispara_tras_cooldown_si_sigue_detectado():
    estado = estado_inicial_celular()
    eventos_celular = []
    t = 0.0
    while t <= 33.0:
        estado, eventos = procesar_objetos(
            estado, CELULAR_CONFIANZA_ALTA, timestamp=t, config=CONFIG
        )
        eventos_celular += [e for e in eventos if e.tipo == "uso_celular"]
        t += 0.5
    assert len(eventos_celular) == 2
    assert eventos_celular[0].valor == 2.5
    assert eventos_celular[1].valor == 32.5


def test_multiples_objetos_uno_es_celular_con_confianza_suficiente():
    estado = estado_inicial_celular()
    objetos = [
        ObjetoDetectado(etiqueta="person", confianza=0.9),
        ObjetoDetectado(etiqueta="cell phone", confianza=0.6),
    ]
    nuevo_estado, _ = procesar_objetos(estado, objetos, timestamp=0.0, config=CONFIG)
    assert nuevo_estado.deteccion_inicio == 0.0


def test_uso_celular_no_dispara_con_valor_inflado_tras_hueco_prolongado():
    # Reproduce el mismo hallazgo ya aplicado a microsueno/distraccion/
    # bostezo: si el modelo no proceso frames durante un tramo largo y
    # luego se retoma con el celular ya detectado, el temporizador NO debe
    # asumir que estuvo detectado desde antes del hueco.
    estado = estado_inicial_celular()
    estado, _ = procesar_objetos(estado, SIN_OBJETOS, timestamp=0.0, config=CONFIG)
    # Hueco prolongado: sin llamadas entre t=0 y t=50.
    estado, eventos = procesar_objetos(
        estado, CELULAR_CONFIANZA_ALTA, timestamp=50.0, config=CONFIG
    )
    assert eventos == []
    assert estado.deteccion_inicio == 50.0
