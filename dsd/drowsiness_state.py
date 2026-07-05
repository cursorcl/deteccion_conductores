from dataclasses import dataclass, field
from typing import List, Optional

from dsd.config import ConfigSomnolencia


@dataclass
class Muestra:
    timestamp: float
    cerrado: bool


@dataclass
class EstadoSomnolencia:
    muestras: List[Muestra] = field(default_factory=list)
    cierre_inicio: Optional[float] = None
    ultimo_disparo_microsueno: Optional[float] = None
    ultimo_disparo_perclos: Optional[float] = None
    primer_timestamp: Optional[float] = None


@dataclass
class EventoSomnolencia:
    tipo: str
    valor: float


def estado_inicial_somnolencia() -> EstadoSomnolencia:
    return EstadoSomnolencia()


def _calcular_perclos(muestras: List[Muestra]) -> float:
    # PERCLOS ponderado por tiempo: cada intervalo entre dos muestras
    # consecutivas se atribuye al estado (cerrado/abierto) de la muestra mas
    # antigua del par. Esto es correcto incluso si el frame rate varia (a
    # diferencia de contar muestras cerradas / muestras totales, que asume
    # implicitamente frame rate constante).
    tiempo_total = 0.0
    tiempo_cerrado = 0.0
    for anterior, siguiente in zip(muestras, muestras[1:]):
        dt = siguiente.timestamp - anterior.timestamp
        tiempo_total += dt
        if anterior.cerrado:
            tiempo_cerrado += dt
    if tiempo_total == 0.0:
        return 0.0
    return tiempo_cerrado / tiempo_total


def procesar_ear(
    estado: EstadoSomnolencia,
    ear: float,
    timestamp: float,
    config: ConfigSomnolencia,
) -> tuple[EstadoSomnolencia, list[EventoSomnolencia]]:
    eventos: List[EventoSomnolencia] = []
    cerrado = ear < config.ear_umbral

    # --- Microsueno: temporizador de cierre continuo ---
    if cerrado:
        cierre_inicio = estado.cierre_inicio if estado.cierre_inicio is not None else timestamp
    else:
        cierre_inicio = None

    duracion_cierre = (timestamp - cierre_inicio) if cierre_inicio is not None else 0.0
    ultimo_disparo_microsueno = estado.ultimo_disparo_microsueno
    if cierre_inicio is not None and duracion_cierre > config.microsueno_segundos:
        en_cooldown = (
            ultimo_disparo_microsueno is not None
            and (timestamp - ultimo_disparo_microsueno) < config.cooldown_segundos
        )
        if not en_cooldown:
            eventos.append(EventoSomnolencia(tipo="microsueno", valor=duracion_cierre))
            ultimo_disparo_microsueno = timestamp

    # --- PERCLOS: ventana deslizante ---
    primer_timestamp = estado.primer_timestamp if estado.primer_timestamp is not None else timestamp
    muestras = [
        m for m in estado.muestras if m.timestamp >= timestamp - config.perclos_ventana_segundos
    ]
    muestras.append(Muestra(timestamp=timestamp, cerrado=cerrado))

    ultimo_disparo_perclos = estado.ultimo_disparo_perclos
    ventana_cubierta = (timestamp - primer_timestamp) >= config.perclos_ventana_segundos
    if ventana_cubierta and len(muestras) >= 2:
        perclos = _calcular_perclos(muestras)
        if perclos >= config.perclos_umbral:
            en_cooldown = (
                ultimo_disparo_perclos is not None
                and (timestamp - ultimo_disparo_perclos) < config.cooldown_segundos
            )
            if not en_cooldown:
                eventos.append(EventoSomnolencia(tipo="perclos", valor=perclos))
                ultimo_disparo_perclos = timestamp

    nuevo_estado = EstadoSomnolencia(
        muestras=muestras,
        cierre_inicio=cierre_inicio,
        ultimo_disparo_microsueno=ultimo_disparo_microsueno,
        ultimo_disparo_perclos=ultimo_disparo_perclos,
        primer_timestamp=primer_timestamp,
    )
    return nuevo_estado, eventos
