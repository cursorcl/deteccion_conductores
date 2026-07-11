from dataclasses import dataclass

import yaml

CAMPOS_REQUERIDOS = (
    "ear_umbral",
    "microsueno_segundos",
    "perclos_ventana_segundos",
    "perclos_umbral",
    "cooldown_segundos",
    "perclos_cobertura_minima",
    "gap_maximo_segundos",
)


@dataclass
class ConfigSomnolencia:
    ear_umbral: float
    microsueno_segundos: float
    perclos_ventana_segundos: float
    perclos_umbral: float
    cooldown_segundos: float
    perclos_cobertura_minima: float
    gap_maximo_segundos: float


def cargar_config(path: str) -> ConfigSomnolencia:
    with open(path, "r", encoding="utf-8") as archivo:
        datos = yaml.safe_load(archivo)

    faltantes = [campo for campo in CAMPOS_REQUERIDOS if campo not in datos]
    if faltantes:
        raise KeyError(
            f"Faltan claves requeridas en el archivo de configuracion '{path}': {faltantes}"
        )

    return ConfigSomnolencia(
        ear_umbral=float(datos["ear_umbral"]),
        microsueno_segundos=float(datos["microsueno_segundos"]),
        perclos_ventana_segundos=float(datos["perclos_ventana_segundos"]),
        perclos_umbral=float(datos["perclos_umbral"]),
        cooldown_segundos=float(datos["cooldown_segundos"]),
        perclos_cobertura_minima=float(datos["perclos_cobertura_minima"]),
        gap_maximo_segundos=float(datos["gap_maximo_segundos"]),
    )


CAMPOS_REQUERIDOS_DISTRACCION = (
    "distraccion_segundos",
    "yaw_umbral_grados",
    "pitch_umbral_grados",
    "gaze_ratio_umbral",
    "cooldown_segundos",
    "gap_maximo_segundos",
)


@dataclass
class ConfigDistraccion:
    distraccion_segundos: float
    yaw_umbral_grados: float
    pitch_umbral_grados: float
    gaze_ratio_umbral: float
    cooldown_segundos: float
    gap_maximo_segundos: float


def cargar_config_distraccion(path: str) -> ConfigDistraccion:
    with open(path, "r", encoding="utf-8") as archivo:
        datos = yaml.safe_load(archivo)

    faltantes = [campo for campo in CAMPOS_REQUERIDOS_DISTRACCION if campo not in datos]
    if faltantes:
        raise KeyError(
            f"Faltan claves requeridas en el archivo de configuracion '{path}': {faltantes}"
        )

    return ConfigDistraccion(
        distraccion_segundos=float(datos["distraccion_segundos"]),
        yaw_umbral_grados=float(datos["yaw_umbral_grados"]),
        pitch_umbral_grados=float(datos["pitch_umbral_grados"]),
        gaze_ratio_umbral=float(datos["gaze_ratio_umbral"]),
        cooldown_segundos=float(datos["cooldown_segundos"]),
        gap_maximo_segundos=float(datos["gap_maximo_segundos"]),
    )
