# Reconocimiento de Conductor + Máquina de Estados de Sesión — Plan de Implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconocer al conductor frente a la cámara del Mac y manejar el ciclo de vida de la sesión de conducción (abrir al reconocer un conductor conocido, cerrar tras 10s de ausencia), persistiendo sesiones en SQLite.

**Architecture:** Aplicación Python de un proceso, dos hilos (cámara/UI + reconocimiento en background con DeepFace, igual patrón que `~/Dev/eosorio/facial-access/recognize_deepface.py`). Módulos aislados y testeables: `session_state.py` (máquina de estados pura), `db.py` (SQLite), `recognition.py` (wrapper de DeepFace), `enroll.py` (CLI de enrolamiento), `main.py` (integración).

**Tech Stack:** Python 3.11.11, OpenCV (`opencv-python`), DeepFace (ArcFace + detector `mtcnn`), `sqlite3` (stdlib), `pytest`.

## Global Constraints

- Umbral de reconocimiento: `UMBRAL_ESTRICTO = 0.68` (distancia ArcFace, menor o igual = match).
- Modelo: `ArcFace`, detector: `mtcnn` (igual que `facial-access/recognize_deepface.py`).
- Timeout de ausencia para cerrar sesión: **estrictamente mayor a 10.0 segundos** (10.0 exacto NO cierra la sesión).
- Si durante una sesión activa aparece otro conductor conocido, la sesión actual **no se interrumpe** — solo cuenta como "conductor actual no visto" en ese instante.
- Carpeta de fotos de conductores: `known_drivers/<nombre>/foto_N.jpg`.
- Base de datos: archivo `data/app.db` (SQLite).
- Versiones de dependencias (confirmadas funcionando en esta Mac en `~/Dev/eosorio/facial-access/.venv`): `deepface==0.0.100`, `opencv-python==5.0.0.93`, `tensorflow==2.21.0`, `numpy==2.4.6`.
- Fuera de alcance: detectores de comportamiento, sync a central, alertas, puerto a Orange Pi.

---

### Task 1: Project setup

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `dsd/__init__.py`
- Create: `conftest.py`

**Interfaces:**
- Produces: paquete Python `dsd` importable desde la raíz del proyecto; entorno virtual `.venv` con dependencias instaladas.

- [ ] **Step 1: Crear estructura de archivos**

`requirements.txt`:
```
opencv-python==5.0.0.93
deepface==0.0.100
tensorflow==2.21.0
numpy==2.4.6
pytest
```

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
known_drivers/
data/
.DS_Store
.pytest_cache/
```

`dsd/__init__.py`: (vacío)

`conftest.py`: (vacío — asegura que pytest agregue la raíz del proyecto a `sys.path` para poder importar `dsd`)

- [ ] **Step 2: Crear entorno virtual e instalar dependencias**

Run:
```bash
cd /Users/cursor/Dev/eosorio/dteccion_somnolencia_distraccion
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Expected: instalación exitosa (puede tardar varios minutos por `tensorflow`/`deepface`).

- [ ] **Step 3: Inicializar git y hacer commit inicial**

Run:
```bash
git init
mkdir -p known_drivers data
git add requirements.txt .gitignore dsd/__init__.py conftest.py docs
git commit -m "chore: project scaffolding"
```
Expected: commit creado, `git log` muestra 1 commit.

---

### Task 2: `db.py` — capa SQLite

**Files:**
- Create: `dsd/db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: nada (módulo base).
- Produces:
  - `init_db(path: str) -> sqlite3.Connection`
  - `crear_conductor(conn: sqlite3.Connection, name: str, created_at: str) -> int`
  - `obtener_conductor_por_nombre(conn: sqlite3.Connection, name: str) -> int | None`
  - `abrir_sesion(conn: sqlite3.Connection, driver_id: int, start_time: str) -> int`
  - `cerrar_sesion(conn: sqlite3.Connection, session_id: int, end_time: str) -> None`

- [ ] **Step 1: Escribir los tests que deben fallar**

`tests/test_db.py`:
```python
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
```

- [ ] **Step 2: Ejecutar los tests y confirmar que fallan**

Run: `source .venv/bin/activate && pytest tests/test_db.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dsd.db'`

- [ ] **Step 3: Implementar `dsd/db.py`**

```python
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
```

- [ ] **Step 4: Ejecutar los tests y confirmar que pasan**

Run: `pytest tests/test_db.py -v`
Expected: PASS — 7 tests verdes.

- [ ] **Step 5: Commit**

```bash
git add dsd/db.py tests/test_db.py
git commit -m "feat: capa SQLite para conductores y sesiones"
```

---

### Task 3: `session_state.py` — máquina de estados pura

**Files:**
- Create: `dsd/session_state.py`
- Test: `tests/test_session_state.py`

**Interfaces:**
- Consumes: nada (módulo puro, sin dependencias de otros módulos del proyecto).
- Produces:
  - `class Estado(Enum)` con miembros `BUSCANDO`, `ACTIVA`
  - `class SessionState` (dataclass) con campos `estado: Estado`, `conductor_actual: Optional[str]`, `ultima_vez_visto: Optional[float]`
  - `class Evento` (dataclass) con campos `tipo: str`, `conductor: str`
  - `estado_inicial() -> SessionState`
  - `procesar_deteccion(estado: SessionState, conductor_detectado: Optional[str], timestamp: float) -> tuple[SessionState, list[Evento]]`
  - `TIMEOUT_AUSENCIA_SEGUNDOS: float = 10.0`

- [ ] **Step 1: Escribir los tests que deben fallar**

`tests/test_session_state.py`:
```python
from dsd.session_state import Estado, Evento, estado_inicial, procesar_deteccion


def test_buscando_sin_deteccion_permanece_buscando():
    estado = estado_inicial()
    nuevo_estado, eventos = procesar_deteccion(estado, None, timestamp=0.0)
    assert nuevo_estado.estado == Estado.BUSCANDO
    assert eventos == []


def test_buscando_con_deteccion_abre_sesion():
    estado = estado_inicial()
    nuevo_estado, eventos = procesar_deteccion(estado, "Juan", timestamp=0.0)
    assert nuevo_estado.estado == Estado.ACTIVA
    assert nuevo_estado.conductor_actual == "Juan"
    assert eventos == [Evento(tipo="sesion_iniciada", conductor="Juan")]


def test_activa_mismo_conductor_actualiza_ultima_vez_visto():
    estado, _ = procesar_deteccion(estado_inicial(), "Juan", timestamp=0.0)
    nuevo_estado, eventos = procesar_deteccion(estado, "Juan", timestamp=5.0)
    assert nuevo_estado.estado == Estado.ACTIVA
    assert nuevo_estado.ultima_vez_visto == 5.0
    assert eventos == []


def test_activa_otro_conductor_no_cierra_sesion():
    estado, _ = procesar_deteccion(estado_inicial(), "Juan", timestamp=0.0)
    nuevo_estado, eventos = procesar_deteccion(estado, "Pedro", timestamp=1.0)
    assert nuevo_estado.estado == Estado.ACTIVA
    assert nuevo_estado.conductor_actual == "Juan"
    assert eventos == []


def test_activa_no_cierra_justo_antes_del_timeout():
    estado, _ = procesar_deteccion(estado_inicial(), "Juan", timestamp=0.0)
    nuevo_estado, eventos = procesar_deteccion(estado, None, timestamp=9.9)
    assert nuevo_estado.estado == Estado.ACTIVA
    assert eventos == []


def test_activa_no_cierra_exactamente_en_el_limite():
    estado, _ = procesar_deteccion(estado_inicial(), "Juan", timestamp=0.0)
    nuevo_estado, eventos = procesar_deteccion(estado, None, timestamp=10.0)
    assert nuevo_estado.estado == Estado.ACTIVA
    assert eventos == []


def test_activa_cierra_sesion_despues_del_timeout():
    estado, _ = procesar_deteccion(estado_inicial(), "Juan", timestamp=0.0)
    nuevo_estado, eventos = procesar_deteccion(estado, None, timestamp=10.1)
    assert nuevo_estado.estado == Estado.BUSCANDO
    assert eventos == [Evento(tipo="sesion_cerrada", conductor="Juan")]


def test_nuevo_conductor_puede_iniciar_sesion_tras_cierre():
    estado, _ = procesar_deteccion(estado_inicial(), "Juan", timestamp=0.0)
    estado, _ = procesar_deteccion(estado, None, timestamp=10.1)
    nuevo_estado, eventos = procesar_deteccion(estado, "Pedro", timestamp=11.0)
    assert nuevo_estado.estado == Estado.ACTIVA
    assert nuevo_estado.conductor_actual == "Pedro"
    assert eventos == [Evento(tipo="sesion_iniciada", conductor="Pedro")]
```

- [ ] **Step 2: Ejecutar los tests y confirmar que fallan**

Run: `pytest tests/test_session_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dsd.session_state'`

- [ ] **Step 3: Implementar `dsd/session_state.py`**

```python
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

TIMEOUT_AUSENCIA_SEGUNDOS = 10.0


class Estado(Enum):
    BUSCANDO = auto()
    ACTIVA = auto()


@dataclass
class SessionState:
    estado: Estado
    conductor_actual: Optional[str] = None
    ultima_vez_visto: Optional[float] = None


@dataclass
class Evento:
    tipo: str
    conductor: str


def estado_inicial() -> SessionState:
    return SessionState(estado=Estado.BUSCANDO)


def procesar_deteccion(
    estado: SessionState,
    conductor_detectado: Optional[str],
    timestamp: float,
) -> tuple[SessionState, list[Evento]]:
    if estado.estado == Estado.BUSCANDO:
        if conductor_detectado is not None:
            nuevo_estado = SessionState(
                estado=Estado.ACTIVA,
                conductor_actual=conductor_detectado,
                ultima_vez_visto=timestamp,
            )
            return nuevo_estado, [Evento(tipo="sesion_iniciada", conductor=conductor_detectado)]
        return estado, []

    # Estado.ACTIVA
    if conductor_detectado == estado.conductor_actual:
        nuevo_estado = SessionState(
            estado=Estado.ACTIVA,
            conductor_actual=estado.conductor_actual,
            ultima_vez_visto=timestamp,
        )
        return nuevo_estado, []

    tiempo_ausente = timestamp - estado.ultima_vez_visto
    if tiempo_ausente > TIMEOUT_AUSENCIA_SEGUNDOS:
        conductor_saliente = estado.conductor_actual
        return estado_inicial(), [Evento(tipo="sesion_cerrada", conductor=conductor_saliente)]

    return estado, []
```

- [ ] **Step 4: Ejecutar los tests y confirmar que pasan**

Run: `pytest tests/test_session_state.py -v`
Expected: PASS — 8 tests verdes.

- [ ] **Step 5: Commit**

```bash
git add dsd/session_state.py tests/test_session_state.py
git commit -m "feat: maquina de estados de sesion de conductor"
```

---

### Task 4: `enroll.py` — CLI de enrolamiento de conductores

**Files:**
- Create: `dsd/enroll.py`

**Interfaces:**
- Consumes: `init_db`, `crear_conductor`, `obtener_conductor_por_nombre` de `dsd.db` (Task 2).
- Produces: comando `python -m dsd.enroll --name <nombre>`; carpeta `known_drivers/<nombre>/foto_N.jpg`; fila en tabla `drivers`.

- [ ] **Step 1: Implementar `dsd/enroll.py`**

```python
import argparse
import os
import sys
from datetime import datetime, timezone

import cv2

from dsd.db import crear_conductor, init_db, obtener_conductor_por_nombre

DIRECTORIO_CONDUCTORES = "known_drivers"
RUTA_DB = "data/app.db"
FOTOS_POR_CONDUCTOR = 5


def enrolar(name: str) -> None:
    carpeta = os.path.join(DIRECTORIO_CONDUCTORES, name)
    os.makedirs(carpeta, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la camara.")
        sys.exit(1)

    print(
        f"Enrolando a '{name}'. Presiona 'c' para capturar cada foto "
        f"({FOTOS_POR_CONDUCTOR} en total), 'q' para cancelar."
    )

    capturadas = 0
    while capturadas < FOTOS_POR_CONDUCTOR:
        ret, frame = cap.read()
        if not ret:
            break

        texto = f"Fotos: {capturadas}/{FOTOS_POR_CONDUCTOR} - 'c' capturar, 'q' salir"
        cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Enrolamiento", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            ruta_foto = os.path.join(carpeta, f"foto_{capturadas + 1}.jpg")
            cv2.imwrite(ruta_foto, frame)
            capturadas += 1
            print(f"Foto guardada: {ruta_foto}")
        elif key == ord("q"):
            print("Enrolamiento cancelado.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(1)

    cap.release()
    cv2.destroyAllWindows()

    conn = init_db(RUTA_DB)
    if obtener_conductor_por_nombre(conn, name) is None:
        crear_conductor(conn, name, datetime.now(timezone.utc).isoformat())
        print(f"Conductor '{name}' registrado en la base de datos.")
    else:
        print(f"Conductor '{name}' ya existia en la base de datos, solo se agregaron fotos.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrola un nuevo conductor conocido.")
    parser.add_argument("--name", required=True, help="Nombre del conductor a enrolar")
    args = parser.parse_args()
    enrolar(args.name)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verificación manual**

Run: `source .venv/bin/activate && python -m dsd.enroll --name "TuNombre"`

Con la ventana de cámara abierta, presiona `c` 5 veces (con tu rostro visible en distintos ángulos/luces) y confirma:
- Se imprimen 5 líneas `Foto guardada: known_drivers/TuNombre/foto_N.jpg`.
- Existen 5 archivos en `known_drivers/TuNombre/`.
- Se imprime `Conductor 'TuNombre' registrado en la base de datos.`
- `sqlite3 data/app.db "SELECT * FROM drivers;"` muestra la fila con tu nombre.

- [ ] **Step 3: Commit**

```bash
git add dsd/enroll.py
git commit -m "feat: CLI de enrolamiento de conductores"
```

---

### Task 5: `recognition.py` — wrapper de DeepFace

**Files:**
- Create: `dsd/recognition.py`

**Interfaces:**
- Consumes: fotos ya existentes en `known_drivers/` (creadas en Task 4).
- Produces: `reconocer_conductor(frame) -> tuple[str, float] | None` (usado por `dsd/main.py` en Task 6).

- [ ] **Step 1: Implementar `dsd/recognition.py`**

```python
import os
from typing import Optional, Tuple

from deepface import DeepFace

DIRECTORIO_CONDUCTORES = "known_drivers"
UMBRAL_ESTRICTO = 0.68
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "mtcnn"


def reconocer_conductor(frame) -> Optional[Tuple[str, float]]:
    try:
        resultados = DeepFace.find(
            img_path=frame,
            db_path=DIRECTORIO_CONDUCTORES,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True,
            silent=True,
            threshold=2.0,
        )
    except Exception:
        return None

    for df in resultados:
        if df.empty:
            continue
        distancia = df["distance"][0]
        if distancia <= UMBRAL_ESTRICTO:
            ruta_identidad = df["identity"][0]
            nombre = ruta_identidad.split(os.path.sep)[-2]
            return nombre, distancia

    return None
```

- [ ] **Step 2: Verificación manual**

Run:
```bash
source .venv/bin/activate
python3 -c "
import cv2
from dsd.recognition import reconocer_conductor
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()
print(reconocer_conductor(frame))
"
```
Con tu rostro (enrolado en Task 4) frente a la cámara, expected: `('TuNombre', <valor menor o igual a 0.68>)`.
Repite tapando tu rostro o con otra persona sin enrolar, expected: `None`.

- [ ] **Step 3: Commit**

```bash
git add dsd/recognition.py
git commit -m "feat: reconocimiento de conductor con DeepFace"
```

---

### Task 6: `main.py` — integración completa

**Files:**
- Create: `dsd/main.py`

**Interfaces:**
- Consumes: `reconocer_conductor` (Task 5), `estado_inicial`/`procesar_deteccion`/`Estado` (Task 3), `init_db`/`abrir_sesion`/`cerrar_sesion`/`obtener_conductor_por_nombre` (Task 2).
- Produces: comando `python -m dsd.main` — aplicación completa con ventana de video, overlay de estado y persistencia de sesiones.

- [ ] **Step 1: Implementar `dsd/main.py`**

```python
import threading
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

import cv2

from dsd.db import abrir_sesion, cerrar_sesion, init_db, obtener_conductor_por_nombre
from dsd.recognition import reconocer_conductor
from dsd.session_state import Estado, estado_inicial, procesar_deteccion

RUTA_DB = "data/app.db"

frame_actual = None
resultado_cacheado: Optional[Tuple[str, float]] = None
lock = threading.Lock()
detener = threading.Event()


def hilo_reconocimiento() -> None:
    global resultado_cacheado
    while not detener.is_set():
        with lock:
            frame = frame_actual.copy() if frame_actual is not None else None
        if frame is None:
            continue
        resultado = reconocer_conductor(frame)
        with lock:
            resultado_cacheado = resultado


def main() -> None:
    global frame_actual

    conn = init_db(RUTA_DB)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la camara.")
        return

    hilo = threading.Thread(target=hilo_reconocimiento, daemon=True)
    hilo.start()

    estado = estado_inicial()
    session_id_activo = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            with lock:
                frame_actual = frame.copy()
                resultado = resultado_cacheado

            nombre_detectado = resultado[0] if resultado else None
            timestamp = time.monotonic()
            estado, eventos = procesar_deteccion(estado, nombre_detectado, timestamp)

            for evento in eventos:
                ahora_iso = datetime.now(timezone.utc).isoformat()
                if evento.tipo == "sesion_iniciada":
                    driver_id = obtener_conductor_por_nombre(conn, evento.conductor)
                    session_id_activo = abrir_sesion(conn, driver_id, ahora_iso)
                    print(f"Sesion iniciada: {evento.conductor}")
                elif evento.tipo == "sesion_cerrada":
                    cerrar_sesion(conn, session_id_activo, ahora_iso)
                    print(f"Sesion cerrada: {evento.conductor}")
                    session_id_activo = None

            if estado.estado == Estado.ACTIVA:
                texto = f"Sesion activa: {estado.conductor_actual}"
            else:
                texto = "Buscando conductor..."

            cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Deteccion de conductor", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detener.set()
        hilo.join(timeout=2)
        if session_id_activo is not None:
            cerrar_sesion(conn, session_id_activo, datetime.now(timezone.utc).isoformat())
            print("Sesion activa cerrada al salir.")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verificación manual — escenarios completos**

Run: `source .venv/bin/activate && python -m dsd.main`

Con el conductor enrolado en Task 4, verificar en orden:
1. Mostrar tu rostro → consola imprime `Sesion iniciada: TuNombre`, overlay muestra `Sesion activa: TuNombre`.
2. Salir del cuadro por ~5 segundos y volver a entrar → la consola **no** debe imprimir `Sesion cerrada`, la sesión sigue activa.
3. Salir del cuadro por más de 10 segundos → consola imprime `Sesion cerrada: TuNombre`, overlay vuelve a `Buscando conductor...`.
4. Volver a mostrar tu rostro → se abre una nueva sesión (`Sesion iniciada: TuNombre` de nuevo).
5. Presionar `q` con una sesión activa → consola imprime `Sesion activa cerrada al salir.`

- [ ] **Step 3: Verificar persistencia en la base de datos**

Run: `sqlite3 data/app.db "SELECT id, driver_id, start_time, end_time FROM sessions ORDER BY id;"`
Expected: cada sesión abierta en el paso anterior aparece con su `start_time`, y todas tienen `end_time` distinto de `NULL` (ninguna quedó "colgada").

- [ ] **Step 4: Commit**

```bash
git add dsd/main.py
git commit -m "feat: integracion de reconocimiento y maquina de estados de sesion"
```

---

## Self-Review

- **Cobertura del spec:** arquitectura multi-hilo (Task 6), `recognition.py` (Task 5), `session_state.py` con las 4 reglas de transición (Task 3, 8 tests), `db.py` con el esquema exacto de `drivers`/`sessions` (Task 2), `enroll.py` con captura por teclado y carpeta `known_drivers/<name>/` (Task 4), manejo de sesión colgada al salir (Task 6 Step 1 `finally`), cámara no disponible → error claro sin reintentos (Task 4 y 6). Todo el alcance del spec está cubierto; nada del spec quedó sin tarea.
- **Placeholders:** ninguno — todos los pasos tienen código completo o comandos exactos con salida esperada.
- **Consistencia de tipos/nombres:** `procesar_deteccion` y `Evento`/`Estado` se usan igual en Task 3 y Task 6; `reconocer_conductor` se usa igual en Task 5 y Task 6; `init_db`/`abrir_sesion`/`cerrar_sesion`/`obtener_conductor_por_nombre` se usan igual en Task 2, 4 y 6. Verificado sin discrepancias.
