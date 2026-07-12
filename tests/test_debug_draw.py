from dsd.debug_draw import CONEXIONES_TESELACION


def test_conexiones_teselacion_no_esta_vacia():
    assert len(CONEXIONES_TESELACION) > 0


def test_conexiones_teselacion_indices_en_rango_valido():
    # Los 478 landmarks de Mediapipe van de indice 0 a 477; si una version
    # futura de Mediapipe cambia esta tabla, este test debe fallar en vez
    # de dejar pasar un indice fuera de rango silenciosamente (que causaria
    # un IndexError recien al dibujar con una camara real).
    for conexion in CONEXIONES_TESELACION:
        assert 0 <= conexion.start <= 477
        assert 0 <= conexion.end <= 477
