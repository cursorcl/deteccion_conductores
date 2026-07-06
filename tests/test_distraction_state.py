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


def test_frente_sin_giro_ni_desviacion_no_dispara_nada():
    estado = estado_inicial_distraccion()
    nuevo_estado, eventos = procesar_pose_y_mirada(
        estado, YAW_FRENTE, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
        timestamp=0.0, config=CONFIG,
    )
    assert eventos == []
    assert nuevo_estado.cabeza_girada_inicio is None
    assert nuevo_estado.mirada_desviada_inicio is None


def test_giro_breve_no_dispara_distraccion_cabeza():
    estado = estado_inicial_distraccion()
    for t in [0.0, 1.0]:
        estado, eventos = procesar_pose_y_mirada(
            estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
            timestamp=t, config=CONFIG,
        )
        assert eventos == []


def test_giro_exactamente_en_el_limite_no_dispara():
    estado = estado_inicial_distraccion()
    for t in [0.0, 2.0]:
        estado, eventos = procesar_pose_y_mirada(
            estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
            timestamp=t, config=CONFIG,
        )
    assert eventos == []


def test_giro_sostenido_dispara_distraccion_cabeza():
    estado = estado_inicial_distraccion()
    for t in [0.0, 1.0, 2.0]:
        estado, _ = procesar_pose_y_mirada(
            estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
            timestamp=t, config=CONFIG,
        )
    estado, eventos = procesar_pose_y_mirada(
        estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
        timestamp=2.1, config=CONFIG,
    )
    assert eventos == [EventoDistraccion(tipo="distraccion_cabeza", valor=2.1)]


def test_volver_a_mirar_al_frente_reinicia_temporizador_cabeza():
    estado = estado_inicial_distraccion()
    estado, _ = procesar_pose_y_mirada(
        estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
        timestamp=0.0, config=CONFIG,
    )
    estado, _ = procesar_pose_y_mirada(
        estado, YAW_FRENTE, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
        timestamp=0.5, config=CONFIG,
    )
    assert estado.cabeza_girada_inicio is None
    estado, _ = procesar_pose_y_mirada(
        estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
        timestamp=0.6, config=CONFIG,
    )
    assert estado.cabeza_girada_inicio == 0.6


def test_distraccion_cabeza_no_re_dispara_dentro_del_cooldown():
    estado = estado_inicial_distraccion()
    for t in [0.0, 2.1]:
        estado, eventos = procesar_pose_y_mirada(
            estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
            timestamp=t, config=CONFIG,
        )
    assert eventos == [EventoDistraccion(tipo="distraccion_cabeza", valor=2.1)]
    estado, eventos = procesar_pose_y_mirada(
        estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
        timestamp=20.0, config=CONFIG,
    )
    assert eventos == []
    estado, eventos = procesar_pose_y_mirada(
        estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
        timestamp=31.5, config=CONFIG,
    )
    assert eventos == []


def test_distraccion_cabeza_re_dispara_tras_cooldown_si_sigue_girada():
    estado = estado_inicial_distraccion()
    for t in [0.0, 2.1, 20.0, 31.5]:
        estado, eventos = procesar_pose_y_mirada(
            estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
            timestamp=t, config=CONFIG,
        )
    # El ultimo disparo de cabeza fue en t=2.1 (cabeza_girada_inicio=0.0);
    # su cooldown de 30s expira en t=32.1, asi que el llamado final debe
    # ser posterior a ese punto para volver a disparar.
    estado, eventos = procesar_pose_y_mirada(
        estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
        timestamp=32.2, config=CONFIG,
    )
    assert eventos == [EventoDistraccion(tipo="distraccion_cabeza", valor=32.2)]


def test_mirada_breve_no_dispara_distraccion_mirada():
    estado = estado_inicial_distraccion()
    for t in [0.0, 1.0]:
        estado, eventos = procesar_pose_y_mirada(
            estado, YAW_FRENTE, PITCH_FRENTE, GAZE_DESVIADO, GAZE_CENTRADO,
            timestamp=t, config=CONFIG,
        )
        assert eventos == []


def test_mirada_sostenida_dispara_distraccion_mirada():
    estado = estado_inicial_distraccion()
    for t in [0.0, 1.0, 2.0]:
        estado, _ = procesar_pose_y_mirada(
            estado, YAW_FRENTE, PITCH_FRENTE, GAZE_DESVIADO, GAZE_CENTRADO,
            timestamp=t, config=CONFIG,
        )
    estado, eventos = procesar_pose_y_mirada(
        estado, YAW_FRENTE, PITCH_FRENTE, GAZE_DESVIADO, GAZE_CENTRADO,
        timestamp=2.1, config=CONFIG,
    )
    assert eventos == [EventoDistraccion(tipo="distraccion_mirada", valor=2.1)]


def test_distraccion_mirada_no_re_dispara_dentro_del_cooldown():
    estado = estado_inicial_distraccion()
    for t in [0.0, 2.1]:
        estado, eventos = procesar_pose_y_mirada(
            estado, YAW_FRENTE, PITCH_FRENTE, GAZE_DESVIADO, GAZE_CENTRADO,
            timestamp=t, config=CONFIG,
        )
    assert eventos == [EventoDistraccion(tipo="distraccion_mirada", valor=2.1)]
    estado, eventos = procesar_pose_y_mirada(
        estado, YAW_FRENTE, PITCH_FRENTE, GAZE_DESVIADO, GAZE_CENTRADO,
        timestamp=20.0, config=CONFIG,
    )
    assert eventos == []


def test_distraccion_mirada_re_dispara_tras_cooldown_si_sigue_desviada():
    estado = estado_inicial_distraccion()
    for t in [0.0, 2.1, 20.0, 31.5]:
        estado, eventos = procesar_pose_y_mirada(
            estado, YAW_FRENTE, PITCH_FRENTE, GAZE_DESVIADO, GAZE_CENTRADO,
            timestamp=t, config=CONFIG,
        )
    estado, eventos = procesar_pose_y_mirada(
        estado, YAW_FRENTE, PITCH_FRENTE, GAZE_DESVIADO, GAZE_CENTRADO,
        timestamp=32.2, config=CONFIG,
    )
    assert eventos == [EventoDistraccion(tipo="distraccion_mirada", valor=32.2)]


def test_cabeza_y_mirada_son_independientes_entre_si():
    estado = estado_inicial_distraccion()
    for t in [0.0, 1.0, 2.0]:
        estado, _ = procesar_pose_y_mirada(
            estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
            timestamp=t, config=CONFIG,
        )
    estado, eventos = procesar_pose_y_mirada(
        estado, YAW_GIRADA, PITCH_FRENTE, GAZE_CENTRADO, GAZE_CENTRADO,
        timestamp=2.1, config=CONFIG,
    )
    assert eventos == [EventoDistraccion(tipo="distraccion_cabeza", valor=2.1)]
    assert estado.mirada_desviada_inicio is None
    assert estado.ultimo_disparo_mirada is None


def test_cabeza_y_mirada_pueden_dispararse_en_el_mismo_llamado():
    estado = estado_inicial_distraccion()
    for t in [0.0, 1.0, 2.0]:
        estado, _ = procesar_pose_y_mirada(
            estado, YAW_GIRADA, PITCH_FRENTE, GAZE_DESVIADO, GAZE_CENTRADO,
            timestamp=t, config=CONFIG,
        )
    estado, eventos = procesar_pose_y_mirada(
        estado, YAW_GIRADA, PITCH_FRENTE, GAZE_DESVIADO, GAZE_CENTRADO,
        timestamp=2.1, config=CONFIG,
    )
    tipos = {e.tipo for e in eventos}
    assert tipos == {"distraccion_cabeza", "distraccion_mirada"}


def test_solo_pitch_alto_tambien_activa_cabeza_girada():
    estado = estado_inicial_distraccion()
    estado, _ = procesar_pose_y_mirada(
        estado, YAW_FRENTE, 20.0, GAZE_CENTRADO, GAZE_CENTRADO,
        timestamp=0.0, config=CONFIG,
    )
    assert estado.cabeza_girada_inicio == 0.0


def test_solo_componente_vertical_de_mirada_tambien_activa_mirada_desviada():
    estado = estado_inicial_distraccion()
    estado, _ = procesar_pose_y_mirada(
        estado, YAW_FRENTE, PITCH_FRENTE, GAZE_CENTRADO, 0.75,
        timestamp=0.0, config=CONFIG,
    )
    assert estado.mirada_desviada_inicio == 0.0
