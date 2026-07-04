# Reconocimiento de conductor + máquina de estados de sesión (prototipo Mac)

## Contexto

Este es el primer sub-proyecto de implementación del sistema de detección de somnolencia/distracción del conductor (ver `docs/superpowers/specs/` para futuros sub-proyectos, y el documento de definición de componentes HW/SW previo). El objetivo final es un dispositivo embebido (Orange Pi 5 Plus), pero esta etapa se desarrolla y valida primero en una Mac (Apple M4 Max Pro, 64GB RAM) para iterar rápido, reusando el código y aprendizaje ya existente en `~/Dev/eosorio/facial-access/recognize_deepface.py`.

Alcance de esta etapa: reconocer al conductor frente a la cámara y manejar el ciclo de vida de una "sesión de conducción" (abrir sesión al reconocer un conductor conocido, cerrarla si el conductor está ausente de la cámara por más de 10 segundos). **No incluye** todavía los detectores de comportamiento (somnolencia, distracción, celular, cigarro) ni la sincronización con la central — esos son sub-proyectos futuros.

## Arquitectura

Aplicación Python de un solo proceso, multi-hilo (mismo patrón que `recognize_deepface.py`, ya validado para evitar que DeepFace congele el video):
- **Hilo de cámara/UI**: captura frames de la webcam del Mac y dibuja una ventana OpenCV con overlay (bounding box del rostro, nombre reconocido, y estado de la sesión con duración).
- **Hilo de reconocimiento**: corre `DeepFace.find()` (ArcFace + detector `mtcnn`, umbral de distancia 0.68 — igual que el código existente) de forma continua en segundo plano sobre el frame más reciente, y publica el resultado (nombre reconocido o ninguno) con su timestamp a la máquina de estados.

## Componentes

Cada uno es un módulo independiente y testeable por separado:

1. **`recognition.py`** — envuelve DeepFace. Dado un frame, retorna `(nombre, distancia)` o `None` si no hay match por debajo del umbral. No sabe nada de sesiones ni hilos de cámara.
2. **`session_state.py`** — máquina de estados pura (sin cámara, sin DB, sin reloj real — recibe el tiempo como parámetro para poder testear con TDD). Estados:
   - `BUSCANDO`: sin conductor activo. Al recibir un evento de reconocimiento con un conductor conocido → transiciona a `ACTIVA(conductor=X)` y emite evento `sesion_iniciada`.
   - `ACTIVA(conductor=A)`: guarda `last_seen[A]`. Cada evento de reconocimiento que confirma a A actualiza `last_seen[A]`. Un evento que reconoce a otro conductor conocido (B) o que no reconoce a nadie **no cambia el estado** — solo cuenta como "A no visto" en ese instante. Si `ahora - last_seen[A] > 10s` → transiciona a `BUSCANDO` y emite evento `sesion_cerrada`.
   - Expone una función pura `procesar_evento(estado_actual, deteccion, timestamp) -> (nuevo_estado, eventos_emitidos)` para que sea trivial de testear.
3. **`db.py`** — SQLite (módulo estándar `sqlite3`, sin ORM). Esquema:
   - `drivers(id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, created_at TEXT NOT NULL)`
   - `sessions(id INTEGER PRIMARY KEY, driver_id INTEGER NOT NULL REFERENCES drivers(id), start_time TEXT NOT NULL, end_time TEXT)` — `end_time` es `NULL` mientras la sesión está activa.
   - Funciones: `crear_conductor(name)`, `abrir_sesion(driver_id, start_time)`, `cerrar_sesion(session_id, end_time)`.
4. **`enroll.py`** (CLI) — abre la cámara, muestra preview en vivo; el usuario presiona `c` para capturar cada foto (5 fotos por conductor) y `q` para cancelar. Guarda las fotos en `known_drivers/<name>/foto_N.jpg` (misma convención de carpetas que `facial-access`, que es lo que indexa `DeepFace.find`) e inserta el conductor en la tabla `drivers` si no existe.
5. **`main.py`** — punto de entrada: inicializa cámara, hilo de reconocimiento, máquina de estados y DB; conecta los eventos de la máquina de estados con `abrir_sesion`/`cerrar_sesion`; dibuja el overlay con el estado actual.

## Datos y umbrales (reusados de código existente)

- Modelo: `ArcFace`, detector: `mtcnn`, `UMBRAL_ESTRICTO = 0.68` (igual que `facial-access/recognize_deepface.py`).
- Timeout de ausencia: 10 segundos, configurable como constante.
- Carpeta de enrolamiento: `known_drivers/` (nueva para este proyecto, no se comparte con `facial-access/base_datos_familia`).
- Base de datos: archivo `data/app.db` (SQLite).

## Manejo de errores

- Si la cámara no está disponible al iniciar, `main.py` termina con un mensaje claro (no reintentos silenciosos).
- Si `DeepFace.find()` lanza una excepción sobre un frame (ruido, sin rostro detectable), se ignora ese ciclo y se trata como "nadie reconocido" para ese instante — igual tolerancia que el código existente.
- Si al cerrar la aplicación (Ctrl+C o `q`) hay una sesión activa, se cierra explícitamente en la BD con el timestamp actual antes de salir (no debe quedar una sesión "colgada" sin `end_time`).

## Testing

- `session_state.py` y `db.py` se implementan con TDD (pytest): la máquina de estados se testea con tiempos y eventos simulados (sin cámara real ni DeepFace), cubriendo: apertura de sesión, mantenimiento de sesión pese a otro conductor visible, cierre por timeout exacto al límite de 10s, y no-cierre justo antes del límite.
- `recognition.py`, `enroll.py` y `main.py` dependen de la cámara/DeepFace real y se verifican manualmente corriendo la app (no hay mocks de cámara en esta etapa).

## Fuera de alcance (queda para sub-proyectos futuros)

- Detectores de comportamiento (somnolencia, distracción, celular, cigarro).
- Sincronización con la central (WiFi/celular).
- Alertas en cabina (audio/visual).
- Puerto a Orange Pi 5 Plus / RKNN.
