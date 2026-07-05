from dsd.session_state import Estado, Evento, estado_inicial, procesar_deteccion


def test_buscando_sin_deteccion_permanece_buscando():
    estado = estado_inicial()
    nuevo_estado, eventos = procesar_deteccion(estado, None, timestamp=0.0)
    assert nuevo_estado.estado == Estado.BUSCANDO
    assert eventos == []


def test_buscando_con_deteccion_abre_sesion():
    estado = estado_inicial()
    nuevo_estado, eventos = procesar_deteccion(estado, "Juan", timestamp=0.0)
    assert nuevo_estado.estado == Estado.ACTIVA
    assert nuevo_estado.conductor_actual == "Juan"
    assert eventos == [Evento(tipo="sesion_iniciada", conductor="Juan")]


def test_activa_mismo_conductor_actualiza_ultima_vez_visto():
    estado, _ = procesar_deteccion(estado_inicial(), "Juan", timestamp=0.0)
    nuevo_estado, eventos = procesar_deteccion(estado, "Juan", timestamp=5.0)
    assert nuevo_estado.estado == Estado.ACTIVA
    assert nuevo_estado.ultima_vez_visto == 5.0
    assert eventos == []


def test_activa_otro_conductor_no_cierra_sesion():
    estado, _ = procesar_deteccion(estado_inicial(), "Juan", timestamp=0.0)
    nuevo_estado, eventos = procesar_deteccion(estado, "Pedro", timestamp=1.0)
    assert nuevo_estado.estado == Estado.ACTIVA
    assert nuevo_estado.conductor_actual == "Juan"
    assert eventos == []


def test_activa_otro_conductor_no_avanza_ultima_vez_visto():
    estado, _ = procesar_deteccion(estado_inicial(), "Juan", timestamp=0.0)
    nuevo_estado, _ = procesar_deteccion(estado, "Pedro", timestamp=1.0)
    assert nuevo_estado.ultima_vez_visto == 0.0


def test_activa_ausencia_dentro_del_timeout_no_avanza_ultima_vez_visto():
    estado, _ = procesar_deteccion(estado_inicial(), "Juan", timestamp=0.0)
    nuevo_estado, _ = procesar_deteccion(estado, None, timestamp=9.9)
    assert nuevo_estado.ultima_vez_visto == 0.0


def test_activa_ausencia_prolongada_con_otro_conductor_cierra_sesion():
    estado, _ = procesar_deteccion(estado_inicial(), "Juan", timestamp=0.0)
    nuevo_estado, eventos = procesar_deteccion(estado, "Pedro", timestamp=10.1)
    assert nuevo_estado.estado == Estado.BUSCANDO
    assert eventos == [Evento(tipo="sesion_cerrada", conductor="Juan")]


def test_activa_no_cierra_justo_antes_del_timeout():
    estado, _ = procesar_deteccion(estado_inicial(), "Juan", timestamp=0.0)
    nuevo_estado, eventos = procesar_deteccion(estado, None, timestamp=9.9)
    assert nuevo_estado.estado == Estado.ACTIVA
    assert eventos == []


def test_activa_no_cierra_exactamente_en_el_limite():
    estado, _ = procesar_deteccion(estado_inicial(), "Juan", timestamp=0.0)
    nuevo_estado, eventos = procesar_deteccion(estado, None, timestamp=10.0)
    assert nuevo_estado.estado == Estado.ACTIVA
    assert eventos == []


def test_activa_cierra_sesion_despues_del_timeout():
    estado, _ = procesar_deteccion(estado_inicial(), "Juan", timestamp=0.0)
    nuevo_estado, eventos = procesar_deteccion(estado, None, timestamp=10.1)
    assert nuevo_estado.estado == Estado.BUSCANDO
    assert eventos == [Evento(tipo="sesion_cerrada", conductor="Juan")]


def test_nuevo_conductor_puede_iniciar_sesion_tras_cierre():
    estado, _ = procesar_deteccion(estado_inicial(), "Juan", timestamp=0.0)
    estado, _ = procesar_deteccion(estado, None, timestamp=10.1)
    nuevo_estado, eventos = procesar_deteccion(estado, "Pedro", timestamp=11.0)
    assert nuevo_estado.estado == Estado.ACTIVA
    assert nuevo_estado.conductor_actual == "Pedro"
    assert eventos == [Evento(tipo="sesion_iniciada", conductor="Pedro")]
