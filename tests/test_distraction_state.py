from dsd.config import ConfigDistraccion
from dsd.distraction_state import (
    EventoDistraccion,
    estado_inicial_distraccion,
    procesar_pose_y_mirada,
)

CONFIG = ConfigDistraccion(
    distraccion_segundos=2.0,
    yaw_umbral_grados=20.0,
    pitch_umbral_grados=15.0,
    gaze_ratio_umbral=0.20,
    cooldown_segundos=30.0,
    gap_maximo_segundos=1.0,
)

# Valores de conveniencia: "al frente" no dispara ninguna senal; "girada"
# supera el umbral de yaw; "desviada" supera el umbral de gaze horizontal.
YAW_FRENTE = 0.0
YAW_GIRADA = 25.0
PITCH_FRENTE = 0.0
GAZE_CENTRADO = 0.5
GAZE_DESVIADO = 0.75


def test_estado_inicial_no_tiene_temporizadores_activos():
    estado = estado_inicial_distraccion()
    assert estado.cabeza_girada_inicio is None
    assert estado.mirada_desviada_inicio is None
    assert estado.ultimo_disparo_cabeza is None
    assert estado.ultimo_disparo_mirada is None
    assert estado.ultimo_procesado is None


def test_frente_sin_giro_ni_desviacion_no_dispara_nada():
    estado = estado_inicial_distraccion()
    nuevo_estado, eventos = procesar_pose_y_mirada(
        estado, YAW_FRENTE, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO, True,
        timestamp=0.0, config=CONFIG,
    )
    assert eventos == []
    assert nuevo_estado.cabeza_girada_inicio is None
    assert nuevo_estado.mirada_desviada_inicio is None


def test_giro_breve_no_dispara_distraccion_cabeza():
    estado = estado_inicial_distraccion()
    for t in [0.0, 1.0]:
        estado, eventos = procesar_pose_y_mirada(
            estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO, True,
            timestamp=t, config=CONFIG,
        )
        assert eventos == []


def test_giro_exactamente_en_el_limite_no_dispara():
    # Pasos densos (<= gap_maximo_segundos) para que el chequeo de hueco no
    # interfiera con la condicion de limite que este test quiere ejercitar.
    estado = estado_inicial_distraccion()
    for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
        estado, eventos = procesar_pose_y_mirada(
            estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO, True,
            timestamp=t, config=CONFIG,
        )
    assert eventos == []


def test_giro_sostenido_dispara_distraccion_cabeza():
    estado = estado_inicial_distraccion()
    for t in [0.0, 1.0, 2.0]:
        estado, _ = procesar_pose_y_mirada(
            estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO, True,
            timestamp=t, config=CONFIG,
        )
    estado, eventos = procesar_pose_y_mirada(
        estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO, True,
        timestamp=2.1, config=CONFIG,
    )
    assert eventos == [EventoDistraccion(tipo="distraccion_cabeza", valor=2.1)]


def test_volver_a_mirar_al_frente_reinicia_temporizador_cabeza():
    estado = estado_inicial_distraccion()
    estado, _ = procesar_pose_y_mirada(
        estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO, True,
        timestamp=0.0, config=CONFIG,
    )
    estado, _ = procesar_pose_y_mirada(
        estado, YAW_FRENTE, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO, True,
        timestamp=0.5, config=CONFIG,
    )
    assert estado.cabeza_girada_inicio is None
    estado, _ = procesar_pose_y_mirada(
        estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO, True,
        timestamp=0.6, config=CONFIG,
    )
    assert estado.cabeza_girada_inicio == 0.6


def test_distraccion_cabeza_no_re_dispara_dentro_del_cooldown():
    # Muestreo continuo (paso 0.5s, bajo gap_maximo_segundos) para simular
    # cabeza girada de forma ininterrumpida durante todo el cooldown.
    estado = estado_inicial_distraccion()
    eventos_cabeza = []
    t = 0.0
    while t <= 31.5:
        estado, eventos = procesar_pose_y_mirada(
            estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO, True,
            timestamp=t, config=CONFIG,
        )
        eventos_cabeza += [e for e in eventos if e.tipo == "distraccion_cabeza"]
        t += 0.5
    assert len(eventos_cabeza) == 1
    assert eventos_cabeza[0].valor == 2.5


def test_distraccion_cabeza_re_dispara_tras_cooldown_si_sigue_girada():
    estado = estado_inicial_distraccion()
    eventos_cabeza = []
    t = 0.0
    while t <= 33.0:
        estado, eventos = procesar_pose_y_mirada(
            estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO, True,
            timestamp=t, config=CONFIG,
        )
        eventos_cabeza += [e for e in eventos if e.tipo == "distraccion_cabeza"]
        t += 0.5
    assert len(eventos_cabeza) == 2
    assert eventos_cabeza[0].valor == 2.5
    assert eventos_cabeza[1].valor == 32.5


def test_mirada_breve_no_dispara_distraccion_mirada():
    estado = estado_inicial_distraccion()
    for t in [0.0, 1.0]:
        estado, eventos = procesar_pose_y_mirada(
            estado, YAW_FRENTE, PITCH_FRENTE, GAZE_DESVIADO, GAZE_CENTRADO, True,
            timestamp=t, config=CONFIG,
        )
        assert eventos == []


def test_mirada_sostenida_dispara_distraccion_mirada():
    estado = estado_inicial_distraccion()
    for t in [0.0, 1.0, 2.0]:
        estado, _ = procesar_pose_y_mirada(
            estado, YAW_FRENTE, PITCH_FRENTE, GAZE_DESVIADO, GAZE_CENTRADO, True,
            timestamp=t, config=CONFIG,
        )
    estado, eventos = procesar_pose_y_mirada(
        estado, YAW_FRENTE, PITCH_FRENTE, GAZE_DESVIADO, GAZE_CENTRADO, True,
        timestamp=2.1, config=CONFIG,
    )
    assert eventos == [EventoDistraccion(tipo="distraccion_mirada", valor=2.1)]


def test_distraccion_mirada_no_re_dispara_dentro_del_cooldown():
    estado = estado_inicial_distraccion()
    eventos_mirada = []
    t = 0.0
    while t <= 31.5:
        estado, eventos = procesar_pose_y_mirada(
            estado, YAW_FRENTE, PITCH_FRENTE, GAZE_DESVIADO, GAZE_CENTRADO, True,
            timestamp=t, config=CONFIG,
        )
        eventos_mirada += [e for e in eventos if e.tipo == "distraccion_mirada"]
        t += 0.5
    assert len(eventos_mirada) == 1
    assert eventos_mirada[0].valor == 2.5


def test_distraccion_mirada_re_dispara_tras_cooldown_si_sigue_desviada():
    estado = estado_inicial_distraccion()
    eventos_mirada = []
    t = 0.0
    while t <= 33.0:
        estado, eventos = procesar_pose_y_mirada(
            estado, YAW_FRENTE, PITCH_FRENTE, GAZE_DESVIADO, GAZE_CENTRADO, True,
            timestamp=t, config=CONFIG,
        )
        eventos_mirada += [e for e in eventos if e.tipo == "distraccion_mirada"]
        t += 0.5
    assert len(eventos_mirada) == 2
    assert eventos_mirada[0].valor == 2.5
    assert eventos_mirada[1].valor == 32.5


def test_cabeza_y_mirada_son_independientes_entre_si():
    estado = estado_inicial_distraccion()
    for t in [0.0, 1.0, 2.0]:
        estado, _ = procesar_pose_y_mirada(
            estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO, True,
            timestamp=t, config=CONFIG,
        )
    estado, eventos = procesar_pose_y_mirada(
        estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO, True,
        timestamp=2.1, config=CONFIG,
    )
    assert eventos == [EventoDistraccion(tipo="distraccion_cabeza", valor=2.1)]
    assert estado.mirada_desviada_inicio is None
    assert estado.ultimo_disparo_mirada is None


def test_cabeza_y_mirada_pueden_dispararse_en_el_mismo_llamado():
    estado = estado_inicial_distraccion()
    for t in [0.0, 1.0, 2.0]:
        estado, _ = procesar_pose_y_mirada(
            estado, YAW_GIRADA, PITCH_FRENTE, GAZE_DESVIADO, GAZE_CENTRADO, True,
            timestamp=t, config=CONFIG,
        )
    estado, eventos = procesar_pose_y_mirada(
        estado, YAW_GIRADA, PITCH_FRENTE, GAZE_DESVIADO, GAZE_CENTRADO, True,
        timestamp=2.1, config=CONFIG,
    )
    tipos = {e.tipo for e in eventos}
    assert tipos == {"distraccion_cabeza", "distraccion_mirada"}


def test_solo_pitch_alto_tambien_activa_cabeza_girada():
    estado = estado_inicial_distraccion()
    estado, _ = procesar_pose_y_mirada(
        estado, YAW_FRENTE, 20.0, GAZE_CENTRADO, GAZE_CENTRADO, True,
        timestamp=0.0, config=CONFIG,
    )
    assert estado.cabeza_girada_inicio == 0.0


def test_solo_componente_vertical_de_mirada_tambien_activa_mirada_desviada():
    estado = estado_inicial_distraccion()
    estado, _ = procesar_pose_y_mirada(
        estado, YAW_FRENTE, PITCH_FRENTE, GAZE_CENTRADO, 0.75, True,
        timestamp=0.0, config=CONFIG,
    )
    assert estado.mirada_desviada_inicio == 0.0


def test_ojos_cerrados_suprime_la_senal_de_mirada():
    # Reproduce el hallazgo de la revision final: con los ojos cerrados los
    # landmarks de iris no son confiables, asi que aunque el ratio de
    # mirada aparente estar desviado, no debe iniciar (ni sostener) el
    # temporizador de distraccion_mirada.
    estado = estado_inicial_distraccion()
    for t in [0.0, 1.0, 2.0, 2.1]:
        estado, eventos = procesar_pose_y_mirada(
            estado, YAW_FRENTE, PITCH_FRENTE, GAZE_DESVIADO, GAZE_CENTRADO, False,
            timestamp=t, config=CONFIG,
        )
        assert eventos == []
        assert estado.mirada_desviada_inicio is None


def test_distraccion_cabeza_no_dispara_con_valor_inflado_tras_hueco_prolongado():
    # Reproduce el hallazgo de la revision final: si el rostro no se
    # detecta durante un tramo largo y luego se retoma con la cabeza ya
    # girada, el temporizador NO debe asumir que estuvo girada desde antes
    # del hueco.
    estado = estado_inicial_distraccion()
    estado, _ = procesar_pose_y_mirada(
        estado, YAW_FRENTE, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO, True,
        timestamp=0.0, config=CONFIG,
    )
    # Hueco prolongado: sin llamadas entre t=0 y t=50 (rostro no detectado).
    estado, eventos = procesar_pose_y_mirada(
        estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO, True,
        timestamp=50.0, config=CONFIG,
    )
    assert eventos == []
    assert estado.cabeza_girada_inicio == 50.0
