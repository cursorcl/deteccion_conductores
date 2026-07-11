from dataclasses import dataclass
from typing import Optional


@dataclass
class EstadoTemporizadorSostenido:
    inicio: Optional[float] = None
    ultimo_disparo: Optional[float] = None


def procesar_temporizador_sostenido(
    estado: EstadoTemporizadorSostenido,
    condicion_activa: bool,
    hubo_hueco: bool,
    timestamp: float,
    umbral_segundos: float,
    cooldown_segundos: float,
) -> tuple[EstadoTemporizadorSostenido, Optional[float]]:
    """Temporizador de sostenimiento continuo + cooldown, reutilizable por
    cualquier senal binaria (ojos cerrados, cabeza girada, mirada desviada).

    `hubo_hueco` debe ser True cuando el llamador detecto que paso demasiado
    tiempo desde el ultimo frame procesado (p.ej. porque no se detecto
    rostro durante un tramo largo). Sin este chequeo, un "inicio" fijado
    antes de un hueco prolongado se combinaria con el timestamp posterior
    al hueco, produciendo una duracion inflada y un disparo espurio
    inmediato al reanudar la deteccion -- el mismo tipo de falso positivo
    que motivo el chequeo de cobertura minima de PERCLOS.

    Retorna el nuevo estado y, si corresponde, el valor (duracion en
    segundos) del evento disparado (None si no dispara en este llamado).
    """
    if condicion_activa:
        inicio = timestamp if (estado.inicio is None or hubo_hueco) else estado.inicio
    else:
        inicio = None

    duracion = (timestamp - inicio) if inicio is not None else 0.0
    valor_evento: Optional[float] = None
    ultimo_disparo = estado.ultimo_disparo

    if inicio is not None and duracion > umbral_segundos:
        en_cooldown = (
            ultimo_disparo is not None and (timestamp - ultimo_disparo) < cooldown_segundos
        )
        if not en_cooldown:
            valor_evento = duracion
            ultimo_disparo = timestamp

    nuevo_estado = EstadoTemporizadorSostenido(inicio=inicio, ultimo_disparo=ultimo_disparo)
    return nuevo_estado, valor_evento
