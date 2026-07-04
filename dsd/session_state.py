from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

TIMEOUT_AUSENCIA_SEGUNDOS = 10.0


class Estado(Enum):
    BUSCANDO = auto()
    ACTIVA = auto()


@dataclass
class SessionState:
    estado: Estado
    conductor_actual: Optional[str] = None
    ultima_vez_visto: Optional[float] = None


@dataclass
class Evento:
    tipo: str
    conductor: str


def estado_inicial() -> SessionState:
    return SessionState(estado=Estado.BUSCANDO)


def procesar_deteccion(
    estado: SessionState,
    conductor_detectado: Optional[str],
    timestamp: float,
) -> tuple[SessionState, list[Evento]]:
    if estado.estado == Estado.BUSCANDO:
        if conductor_detectado is not None:
            nuevo_estado = SessionState(
                estado=Estado.ACTIVA,
                conductor_actual=conductor_detectado,
                ultima_vez_visto=timestamp,
            )
            return nuevo_estado, [Evento(tipo="sesion_iniciada", conductor=conductor_detectado)]
        return estado, []

    # Estado.ACTIVA
    if conductor_detectado == estado.conductor_actual:
        nuevo_estado = SessionState(
            estado=Estado.ACTIVA,
            conductor_actual=estado.conductor_actual,
            ultima_vez_visto=timestamp,
        )
        return nuevo_estado, []

    tiempo_ausente = timestamp - estado.ultima_vez_visto
    if tiempo_ausente > TIMEOUT_AUSENCIA_SEGUNDOS:
        conductor_saliente = estado.conductor_actual
        return estado_inicial(), [Evento(tipo="sesion_cerrada", conductor=conductor_saliente)]

    return estado, []
