from dataclasses import dataclass, field
from typing import List, Optional

from dsd.config import ConfigSomnolencia
from dsd.sustained_timer import EstadoTemporizadorSostenido, procesar_temporizador_sostenido


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
    ultimo_procesado: Optional[float] = None


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

    # Un hueco (tiempo excesivo desde el ultimo frame procesado, p.ej.
    # porque el rostro no se detecto durante un tramo) invalida el
    # supuesto de que la condicion se mantuvo continuamente durante ese
    # tramo -- ver dsd/sustained_timer.py.
    hubo_hueco = (
        estado.ultimo_procesado is not None
        and (timestamp - estado.ultimo_procesado) > config.gap_maximo_segundos
    )

    # --- Microsueno: temporizador de cierre continuo (helper compartido) ---
    temporizador_microsueno = EstadoTemporizadorSostenido(
        inicio=estado.cierre_inicio, ultimo_disparo=estado.ultimo_disparo_microsueno
    )
    temporizador_microsueno, valor_microsueno = procesar_temporizador_sostenido(
        temporizador_microsueno,
        cerrado,
        hubo_hueco,
        timestamp,
        config.microsueno_segundos,
        config.cooldown_segundos,
    )
    if valor_microsueno is not None:
        eventos.append(EventoSomnolencia(tipo="microsueno", valor=valor_microsueno))

    # --- PERCLOS: ventana deslizante ---
    primer_timestamp = estado.primer_timestamp if estado.primer_timestamp is not None else timestamp
    muestras = [
        m for m in estado.muestras if m.timestamp >= timestamp - config.perclos_ventana_segundos
    ]
    muestras.append(Muestra(timestamp=timestamp, cerrado=cerrado))

    ultimo_disparo_perclos = estado.ultimo_disparo_perclos
    ventana_cubierta = (timestamp - primer_timestamp) >= config.perclos_ventana_segundos
    # Ademas de haber transcurrido el tiempo nominal de la ventana, exige que
    # las muestras retenidas cubran una fraccion minima real de esa ventana.
    # Sin esto, si el rostro no se detecto durante un tramo largo (frames
    # descartados por completo) y luego llegan solo un par de muestras
    # cercanas entre si, "ventana_cubierta" quedaria satisfecho por el
    # tiempo transcurrido desde el inicio de la sesion aunque los datos
    # reales sean minimos, disparando PERCLOS de forma espuria.
    cobertura_real = muestras[-1].timestamp - muestras[0].timestamp if len(muestras) >= 2 else 0.0
    cobertura_suficiente = cobertura_real >= (
        config.perclos_ventana_segundos * config.perclos_cobertura_minima
    )
    if ventana_cubierta and len(muestras) >= 2 and cobertura_suficiente:
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
        cierre_inicio=temporizador_microsueno.inicio,
        ultimo_disparo_microsueno=temporizador_microsueno.ultimo_disparo,
        ultimo_disparo_perclos=ultimo_disparo_perclos,
        primer_timestamp=primer_timestamp,
        ultimo_procesado=timestamp,
    )
    return nuevo_estado, eventos
