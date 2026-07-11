from dataclasses import dataclass
from typing import Optional

from dsd.config import ConfigDistraccion
from dsd.sustained_timer import EstadoTemporizadorSostenido, procesar_temporizador_sostenido


@dataclass
class EstadoDistraccion:
    cabeza_girada_inicio: Optional[float] = None
    mirada_desviada_inicio: Optional[float] = None
    ultimo_disparo_cabeza: Optional[float] = None
    ultimo_disparo_mirada: Optional[float] = None
    ultimo_procesado: Optional[float] = None


@dataclass
class EventoDistraccion:
    tipo: str
    valor: float


def estado_inicial_distraccion() -> EstadoDistraccion:
    return EstadoDistraccion()


def procesar_pose_y_mirada(
    estado: EstadoDistraccion,
    yaw: float,
    pitch: float,
    gaze_horizontal: float,
    gaze_vertical: float,
    ojos_abiertos: bool,
    timestamp: float,
    config: ConfigDistraccion,
) -> tuple[EstadoDistraccion, list[EventoDistraccion]]:
    eventos: list[EventoDistraccion] = []

    # Un hueco (tiempo excesivo desde el ultimo frame procesado, p.ej.
    # porque no se detecto rostro durante un tramo) invalida el supuesto
    # de que la condicion se mantuvo continuamente durante ese tramo -- ver
    # dsd/sustained_timer.py.
    hubo_hueco = (
        estado.ultimo_procesado is not None
        and (timestamp - estado.ultimo_procesado) > config.gap_maximo_segundos
    )

    cabeza_girada = abs(yaw) > config.yaw_umbral_grados or abs(pitch) > config.pitch_umbral_grados
    # Con los ojos cerrados los landmarks de iris no son confiables (el
    # parpado cubre el globo ocular), asi que la senal de mirada solo se
    # evalua con los ojos abiertos -- de lo contrario un microsueno
    # sostenido podria generar de forma espuria un evento de
    # distraccion_mirada sobre datos de iris basura.
    mirada_desviada = ojos_abiertos and (
        abs(gaze_horizontal - 0.5) > config.gaze_ratio_umbral
        or abs(gaze_vertical - 0.5) > config.gaze_ratio_umbral
    )

    temporizador_cabeza = EstadoTemporizadorSostenido(
        inicio=estado.cabeza_girada_inicio, ultimo_disparo=estado.ultimo_disparo_cabeza
    )
    temporizador_cabeza, valor_cabeza = procesar_temporizador_sostenido(
        temporizador_cabeza,
        cabeza_girada,
        hubo_hueco,
        timestamp,
        config.distraccion_segundos,
        config.cooldown_segundos,
    )
    if valor_cabeza is not None:
        eventos.append(EventoDistraccion(tipo="distraccion_cabeza", valor=valor_cabeza))

    temporizador_mirada = EstadoTemporizadorSostenido(
        inicio=estado.mirada_desviada_inicio, ultimo_disparo=estado.ultimo_disparo_mirada
    )
    temporizador_mirada, valor_mirada = procesar_temporizador_sostenido(
        temporizador_mirada,
        mirada_desviada,
        hubo_hueco,
        timestamp,
        config.distraccion_segundos,
        config.cooldown_segundos,
    )
    if valor_mirada is not None:
        eventos.append(EventoDistraccion(tipo="distraccion_mirada", valor=valor_mirada))

    nuevo_estado = EstadoDistraccion(
        cabeza_girada_inicio=temporizador_cabeza.inicio,
        mirada_desviada_inicio=temporizador_mirada.inicio,
        ultimo_disparo_cabeza=temporizador_cabeza.ultimo_disparo,
        ultimo_disparo_mirada=temporizador_mirada.ultimo_disparo,
        ultimo_procesado=timestamp,
    )
    return nuevo_estado, eventos
