import threading
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

import cv2

from dsd.config import cargar_config
from dsd.db import (
    abrir_sesion,
    cerrar_sesion,
    init_db,
    obtener_conductor_por_nombre,
    registrar_evento,
)
from dsd.drowsiness_state import estado_inicial_somnolencia, procesar_ear
from dsd.eye_metrics import calcular_ear
from dsd.face_mesh import detectar_ojos
from dsd.recognition import reconocer_conductor
from dsd.session_state import Estado, estado_inicial, procesar_deteccion

RUTA_DB = "data/app.db"
RUTA_CONFIG_SOMNOLENCIA = "config/somnolencia.yaml"

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
    config_somnolencia = cargar_config(RUTA_CONFIG_SOMNOLENCIA)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la camara.")
        return

    hilo = threading.Thread(target=hilo_reconocimiento, daemon=True)
    hilo.start()

    estado = estado_inicial()
    estado_somnolencia = estado_inicial_somnolencia()
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
                    if driver_id is None:
                        print(f"Advertencia: '{evento.conductor}' no tiene registro en la base de datos, sesion no persistida.")
                        session_id_activo = None
                    else:
                        session_id_activo = abrir_sesion(conn, driver_id, ahora_iso)
                    # Reinicia el rastreo de somnolencia: cada sesion (mismo
                    # conductor u otro) empieza con el temporizador de
                    # microsueno y la ventana de PERCLOS en blanco.
                    estado_somnolencia = estado_inicial_somnolencia()
                    print(f"Sesion iniciada: {evento.conductor}")
                elif evento.tipo == "sesion_cerrada":
                    if session_id_activo is None:
                        print(f"Advertencia: sesion de '{evento.conductor}' no persistida, nada que cerrar en la base de datos.")
                    else:
                        cerrar_sesion(conn, session_id_activo, ahora_iso)
                        print(f"Sesion cerrada: {evento.conductor}")
                    session_id_activo = None

            if estado.estado == Estado.ACTIVA:
                puntos_ojos = detectar_ojos(frame)
                if puntos_ojos is not None:
                    puntos_ojo_derecho, puntos_ojo_izquierdo = puntos_ojos
                    ear_derecho = calcular_ear(puntos_ojo_derecho)
                    ear_izquierdo = calcular_ear(puntos_ojo_izquierdo)
                    ear_promedio = (ear_derecho + ear_izquierdo) / 2
                    estado_somnolencia, eventos_somnolencia = procesar_ear(
                        estado_somnolencia, ear_promedio, timestamp, config_somnolencia
                    )
                    for evento_somnolencia in eventos_somnolencia:
                        ahora_iso = datetime.now(timezone.utc).isoformat()
                        print(
                            f"Evento de somnolencia: {evento_somnolencia.tipo} "
                            f"(valor={evento_somnolencia.valor:.3f})"
                        )
                        if session_id_activo is not None:
                            registrar_evento(
                                conn,
                                session_id_activo,
                                evento_somnolencia.tipo,
                                evento_somnolencia.valor,
                                ahora_iso,
                            )
                        else:
                            print("Advertencia: evento de somnolencia no persistido, no hay sesion activa en la base de datos.")

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
        conn.close()


if __name__ == "__main__":
    main()
