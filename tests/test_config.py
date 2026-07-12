import pytest

from dsd.config import ConfigSomnolencia, cargar_config
from dsd.config import ConfigDistraccion, cargar_config_distraccion

YAML_VALIDO = """
ear_umbral: 0.21
microsueno_segundos: 1.5
perclos_ventana_segundos: 60
perclos_umbral: 0.15
cooldown_segundos: 30
perclos_cobertura_minima: 0.5
gap_maximo_segundos: 1.0
mar_umbral: 0.6
bostezo_min_segundos: 1.5
bostezo_ventana_segundos: 300
bostezo_umbral_cantidad: 3
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
        perclos_cobertura_minima=0.5,
        gap_maximo_segundos=1.0,
        mar_umbral=0.6,
        bostezo_min_segundos=1.5,
        bostezo_ventana_segundos=300.0,
        bostezo_umbral_cantidad=3.0,
    )


def test_cargar_config_convierte_valores_a_float(tmp_path):
    ruta = tmp_path / "somnolencia.yaml"
    ruta.write_text(YAML_VALIDO)

    config = cargar_config(str(ruta))

    assert isinstance(config.perclos_ventana_segundos, float)
    assert isinstance(config.cooldown_segundos, float)
    assert isinstance(config.bostezo_umbral_cantidad, float)


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
    assert config.perclos_cobertura_minima == 0.5
    assert config.gap_maximo_segundos == 1.0
    assert config.mar_umbral == 0.6
    assert config.bostezo_min_segundos == 1.5
    assert config.bostezo_ventana_segundos == 300.0
    assert config.bostezo_umbral_cantidad == 3.0


YAML_VALIDO_DISTRACCION = """
distraccion_segundos: 2.0
yaw_umbral_grados: 20
pitch_umbral_grados: 15
gaze_ratio_umbral: 0.20
cooldown_segundos: 30
gap_maximo_segundos: 1.0
"""


def test_cargar_config_distraccion_retorna_los_valores_correctos(tmp_path):
    ruta = tmp_path / "distraccion.yaml"
    ruta.write_text(YAML_VALIDO_DISTRACCION)

    config = cargar_config_distraccion(str(ruta))

    assert config == ConfigDistraccion(
        distraccion_segundos=2.0,
        yaw_umbral_grados=20.0,
        pitch_umbral_grados=15.0,
        gaze_ratio_umbral=0.20,
        cooldown_segundos=30.0,
        gap_maximo_segundos=1.0,
    )


def test_cargar_config_distraccion_convierte_valores_a_float(tmp_path):
    ruta = tmp_path / "distraccion.yaml"
    ruta.write_text(YAML_VALIDO_DISTRACCION)

    config = cargar_config_distraccion(str(ruta))

    assert isinstance(config.yaw_umbral_grados, float)
    assert isinstance(config.cooldown_segundos, float)


def test_cargar_config_distraccion_clave_faltante_lanza_keyerror(tmp_path):
    ruta = tmp_path / "distraccion.yaml"
    ruta.write_text("distraccion_segundos: 2.0\n")

    with pytest.raises(KeyError):
        cargar_config_distraccion(str(ruta))


def test_cargar_config_distraccion_archivo_real_del_proyecto():
    config = cargar_config_distraccion("config/distraccion.yaml")
    assert config.distraccion_segundos == 2.0
    assert config.yaw_umbral_grados == 20.0
    assert config.pitch_umbral_grados == 15.0
    assert config.gaze_ratio_umbral == 0.20
    assert config.cooldown_segundos == 30.0
    assert config.gap_maximo_segundos == 1.0


from dsd.config import ConfigCelular, cargar_config_celular

YAML_VALIDO_CELULAR = """
confianza_umbral: 0.5
celular_segundos: 2.0
cooldown_segundos: 30
gap_maximo_segundos: 1.0
"""


def test_cargar_config_celular_retorna_los_valores_correctos(tmp_path):
    ruta = tmp_path / "celular.yaml"
    ruta.write_text(YAML_VALIDO_CELULAR)

    config = cargar_config_celular(str(ruta))

    assert config == ConfigCelular(
        confianza_umbral=0.5,
        celular_segundos=2.0,
        cooldown_segundos=30.0,
        gap_maximo_segundos=1.0,
    )


def test_cargar_config_celular_convierte_valores_a_float(tmp_path):
    ruta = tmp_path / "celular.yaml"
    ruta.write_text(YAML_VALIDO_CELULAR)

    config = cargar_config_celular(str(ruta))

    assert isinstance(config.confianza_umbral, float)
    assert isinstance(config.cooldown_segundos, float)


def test_cargar_config_celular_clave_faltante_lanza_keyerror(tmp_path):
    ruta = tmp_path / "celular.yaml"
    ruta.write_text("confianza_umbral: 0.5\n")

    with pytest.raises(KeyError):
        cargar_config_celular(str(ruta))


def test_cargar_config_celular_archivo_real_del_proyecto():
    config = cargar_config_celular("config/celular.yaml")
    assert config.confianza_umbral == 0.5
    assert config.celular_segundos == 2.0
    assert config.cooldown_segundos == 30.0
    assert config.gap_maximo_segundos == 1.0
