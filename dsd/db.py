import sqlite3
from typing import Optional


def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS drivers (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY,
            driver_id INTEGER NOT NULL REFERENCES drivers(id),
            start_time TEXT NOT NULL,
            end_time TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY,
            session_id INTEGER NOT NULL REFERENCES sessions(id),
            tipo TEXT NOT NULL,
            valor REAL NOT NULL,
            timestamp TEXT NOT NULL,
            synced INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.commit()
    return conn


def crear_conductor(conn: sqlite3.Connection, name: str, created_at: str) -> int:
    cursor = conn.execute(
        "INSERT INTO drivers (name, created_at) VALUES (?, ?)", (name, created_at)
    )
    conn.commit()
    return cursor.lastrowid


def obtener_conductor_por_nombre(conn: sqlite3.Connection, name: str) -> Optional[int]:
    row = conn.execute("SELECT id FROM drivers WHERE name = ?", (name,)).fetchone()
    return row[0] if row else None


def abrir_sesion(conn: sqlite3.Connection, driver_id: int, start_time: str) -> int:
    cursor = conn.execute(
        "INSERT INTO sessions (driver_id, start_time, end_time) VALUES (?, ?, NULL)",
        (driver_id, start_time),
    )
    conn.commit()
    return cursor.lastrowid


def cerrar_sesion(conn: sqlite3.Connection, session_id: int, end_time: str) -> None:
    conn.execute("UPDATE sessions SET end_time = ? WHERE id = ?", (end_time, session_id))
    conn.commit()


def registrar_evento(
    conn: sqlite3.Connection,
    session_id: int,
    tipo: str,
    valor: float,
    timestamp: str,
) -> int:
    cursor = conn.execute(
        "INSERT INTO events (session_id, tipo, valor, timestamp, synced) VALUES (?, ?, ?, ?, 0)",
        (session_id, tipo, valor, timestamp),
    )
    conn.commit()
    return cursor.lastrowid
