import sqlite3

import pytest

from dsd.db import (
    abrir_sesion,
    cerrar_sesion,
    crear_conductor,
    init_db,
    obtener_conductor_por_nombre,
)


@pytest.fixture
def conn():
    return init_db(":memory:")


def test_init_db_crea_tablas(conn):
    tablas = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {"drivers", "sessions"} <= tablas


def test_crear_conductor_retorna_id(conn):
    driver_id = crear_conductor(conn, "Juan", "2026-07-04T10:00:00")
    assert isinstance(driver_id, int)


def test_crear_conductor_nombre_duplicado_lanza_error(conn):
    crear_conductor(conn, "Juan", "2026-07-04T10:00:00")
    with pytest.raises(sqlite3.IntegrityError):
        crear_conductor(conn, "Juan", "2026-07-04T10:05:00")


def test_obtener_conductor_por_nombre_encontrado(conn):
    driver_id = crear_conductor(conn, "Juan", "2026-07-04T10:00:00")
    assert obtener_conductor_por_nombre(conn, "Juan") == driver_id


def test_obtener_conductor_por_nombre_no_encontrado(conn):
    assert obtener_conductor_por_nombre(conn, "Desconocido") is None


def test_abrir_sesion_queda_sin_end_time(conn):
    driver_id = crear_conductor(conn, "Juan", "2026-07-04T10:00:00")
    session_id = abrir_sesion(conn, driver_id, "2026-07-04T10:01:00")
    row = conn.execute(
        "SELECT end_time FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    assert row[0] is None


def test_cerrar_sesion_setea_end_time(conn):
    driver_id = crear_conductor(conn, "Juan", "2026-07-04T10:00:00")
    session_id = abrir_sesion(conn, driver_id, "2026-07-04T10:01:00")
    cerrar_sesion(conn, session_id, "2026-07-04T10:15:00")
    row = conn.execute(
        "SELECT end_time FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    assert row[0] == "2026-07-04T10:15:00"
