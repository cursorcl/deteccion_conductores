import pytest

from dsd.config import ConfigSomnolencia, cargar_config

YAML_VALIDO = """
ear_umbral: 0.21
microsueno_segundos: 1.5
perclos_ventana_segundos: 60
perclos_umbral: 0.15
cooldown_segundos: 30
"""


def test_cargar_config_retorna_los_valores_correctos(tmp_path):
    ruta = tmp_path / "somnolencia.yaml"
    ruta.write_text(YAML_VALIDO)

    config = cargar_config(str(ruta))

    assert config == ConfigSomnolencia(
        ear_umbral=0.21,
        microsueno_segundos=1.5,
        perclos_ventana_segundos=60.0,
        perclos_umbral=0.15,
        cooldown_segundos=30.0,
    )


def test_cargar_config_convierte_valores_a_float(tmp_path):
    ruta = tmp_path / "somnolencia.yaml"
    ruta.write_text(YAML_VALIDO)

    config = cargar_config(str(ruta))

    assert isinstance(config.perclos_ventana_segundos, float)
    assert isinstance(config.cooldown_segundos, float)


def test_cargar_config_clave_faltante_lanza_keyerror(tmp_path):
    ruta = tmp_path / "somnolencia.yaml"
    ruta.write_text("ear_umbral: 0.21\n")

    with pytest.raises(KeyError):
        cargar_config(str(ruta))


def test_cargar_config_archivo_real_del_proyecto():
    config = cargar_config("config/somnolencia.yaml")
    assert config.ear_umbral == 0.21
    assert config.microsueno_segundos == 1.5
    assert config.perclos_ventana_segundos == 60.0
    assert config.perclos_umbral == 0.15
    assert config.cooldown_segundos == 30.0
