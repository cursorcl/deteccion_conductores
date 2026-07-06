from dataclasses import dataclass
from typing import Optional

from dsd.config import ConfigDistraccion


@dataclass
class EstadoDistraccion:
    cabeza_girada_inicio: Optional[float] = None
    mirada_desviada_inicio: Optional[float] = None
    ultimo_disparo_cabeza: Optional[float] = None
    ultimo_disparo_mirada: Optional[float] = None


@dataclass
class EventoDistraccion:
    tipo: str
    valor: float


def estado_inicial_distraccion() -> EstadoDistraccion:
    return EstadoDistraccion()


def _procesar_temporizador_sostenido(
    condicion_activa: bool,
    inicio_actual: Optional[float],
    ultimo_disparo: Optional[float],
    timestamp: float,
    umbral_segundos: float,
    cooldown_segundos: float,
    tipo_evento: str,
) -> tuple[Optional[float], Optional[float], Optional[EventoDistraccion]]:
    if condicion_activa:
        inicio = inicio_actual if inicio_actual is not None else timestamp
    else:
        inicio = None

    duracion = (timestamp - inicio) if inicio is not None else 0.0
    evento = None
    if inicio is not None and duracion > umbral_segundos:
        en_cooldown = (
            ultimo_disparo is not None and (timestamp - ultimo_disparo) < cooldown_segundos
        )
        if not en_cooldown:
            evento = EventoDistraccion(tipo=tipo_evento, valor=duracion)
            ultimo_disparo = timestamp

    return inicio, ultimo_disparo, evento


def procesar_pose_y_mirada(
    estado: EstadoDistraccion,
    yaw: float,
    pitch: float,
    gaze_horizontal: float,
    gaze_vertical: float,
    timestamp: float,
    config: ConfigDistraccion,
) -> tuple[EstadoDistraccion, list[EventoDistraccion]]:
    eventos: list[EventoDistraccion] = []

    cabeza_girada = abs(yaw) > config.yaw_umbral_grados or abs(pitch) > config.pitch_umbral_grados
    mirada_desviada = (
        abs(gaze_horizontal - 0.5) > config.gaze_ratio_umbral
        or abs(gaze_vertical - 0.5) > config.gaze_ratio_umbral
    )

    cabeza_girada_inicio, ultimo_disparo_cabeza, evento_cabeza = _procesar_temporizador_sostenido(
        cabeza_girada,
        estado.cabeza_girada_inicio,
        estado.ultimo_disparo_cabeza,
        timestamp,
        config.distraccion_segundos,
        config.cooldown_segundos,
        "distraccion_cabeza",
    )
    if evento_cabeza is not None:
        eventos.append(evento_cabeza)

    mirada_desviada_inicio, ultimo_disparo_mirada, evento_mirada = _procesar_temporizador_sostenido(
        mirada_desviada,
        estado.mirada_desviada_inicio,
        estado.ultimo_disparo_mirada,
        timestamp,
        config.distraccion_segundos,
        config.cooldown_segundos,
        "distraccion_mirada",
    )
    if evento_mirada is not None:
        eventos.append(evento_mirada)

    nuevo_estado = EstadoDistraccion(
        cabeza_girada_inicio=cabeza_girada_inicio,
        mirada_desviada_inicio=mirada_desviada_inicio,
        ultimo_disparo_cabeza=ultimo_disparo_cabeza,
        ultimo_disparo_mirada=ultimo_disparo_mirada,
    )
    return nuevo_estado, eventos
