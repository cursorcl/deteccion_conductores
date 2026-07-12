from dataclasses import dataclass
from typing import List, Optional

from dsd.config import ConfigCelular
from dsd.object_detection import ObjetoDetectado
from dsd.sustained_timer import EstadoTemporizadorSostenido, procesar_temporizador_sostenido

ETIQUETA_CELULAR = "cell phone"


@dataclass
class EstadoCelular:
    deteccion_inicio: Optional[float] = None
    ultimo_disparo: Optional[float] = None
    ultimo_procesado: Optional[float] = None


@dataclass
class EventoCelular:
    tipo: str
    valor: float


def estado_inicial_celular() -> EstadoCelular:
    return EstadoCelular()


def procesar_objetos(
    estado: EstadoCelular,
    objetos: List[ObjetoDetectado],
    timestamp: float,
    config: ConfigCelular,
) -> tuple[EstadoCelular, list[EventoCelular]]:
    eventos: List[EventoCelular] = []

    # Un hueco (tiempo excesivo desde el ultimo frame procesado) invalida
    # el supuesto de que la condicion se mantuvo continuamente durante ese
    # tramo -- ver dsd/sustained_timer.py.
    hubo_hueco = (
        estado.ultimo_procesado is not None
        and (timestamp - estado.ultimo_procesado) > config.gap_maximo_segundos
    )

    celular_detectado = any(
        o.etiqueta == ETIQUETA_CELULAR and o.confianza >= config.confianza_umbral
        for o in objetos
    )

    temporizador = EstadoTemporizadorSostenido(
        inicio=estado.deteccion_inicio, ultimo_disparo=estado.ultimo_disparo
    )
    temporizador, valor = procesar_temporizador_sostenido(
        temporizador,
        celular_detectado,
        hubo_hueco,
        timestamp,
        config.celular_segundos,
        config.cooldown_segundos,
    )
    if valor is not None:
        eventos.append(EventoCelular(tipo="uso_celular", valor=valor))

    nuevo_estado = EstadoCelular(
        deteccion_inicio=temporizador.inicio,
        ultimo_disparo=temporizador.ultimo_disparo,
        ultimo_procesado=timestamp,
    )
    return nuevo_estado, eventos
