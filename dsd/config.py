from dataclasses import dataclass

import yaml

CAMPOS_REQUERIDOS = (
    "ear_umbral",
    "microsueno_segundos",
    "perclos_ventana_segundos",
    "perclos_umbral",
    "cooldown_segundos",
)


@dataclass
class ConfigSomnolencia:
    ear_umbral: float
    microsueno_segundos: float
    perclos_ventana_segundos: float
    perclos_umbral: float
    cooldown_segundos: float


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
    )
