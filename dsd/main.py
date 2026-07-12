import argparse
import threading
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

import cv2

from dsd.config import cargar_config, cargar_config_distraccion
from dsd.db import (
    abrir_sesion,
    cerrar_sesion,
    init_db,
    obtener_conductor_por_nombre,
    registrar_evento,
)
from dsd.debug_draw import dibujar_malla_debug
from dsd.distraction_state import estado_inicial_distraccion, procesar_pose_y_mirada
from dsd.drowsiness_state import estado_inicial_somnolencia, procesar_somnolencia
from dsd.eye_metrics import calcular_ear
from dsd.face_mesh import detectar_landmarks
from dsd.gaze_metrics import calcular_gaze_ratio
from dsd.mouth_metrics import calcular_mar
from dsd.head_pose import calcular_yaw_pitch
from dsd.recognition import reconocer_conductor
from dsd.session_state import Estado, estado_inicial, procesar_deteccion

RUTA_DB = "data/app.db"
RUTA_CONFIG_SOMNOLENCIA = "config/somnolencia.yaml"
RUTA_CONFIG_DISTRACCION = "config/distraccion.yaml"

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


def main(mostrar_malla: bool = False) -> None:
    global frame_actual

    conn = init_db(RUTA_DB)
    config_somnolencia = cargar_config(RUTA_CONFIG_SOMNOLENCIA)
    config_distraccion = cargar_config_distraccion(RUTA_CONFIG_DISTRACCION)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la camara.")
        return

    hilo = threading.Thread(target=hilo_reconocimiento, daemon=True)
    hilo.start()

    estado = estado_inicial()
    estado_somnolencia = estado_inicial_somnolencia()
    estado_distraccion = estado_inicial_distraccion()
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
                    # Reinicia el rastreo de somnolencia y distraccion: cada
                    # sesion (mismo conductor u otro) empieza con los
                    # temporizadores en blanco.
                    estado_somnolencia = estado_inicial_somnolencia()
                    estado_distraccion = estado_inicial_distraccion()
                    print(f"Sesion iniciada: {evento.conductor}")
                elif evento.tipo == "sesion_cerrada":
                    if session_id_activo is None:
                        print(f"Advertencia: sesion de '{evento.conductor}' no persistida, nada que cerrar en la base de datos.")
                    else:
                        cerrar_sesion(conn, session_id_activo, ahora_iso)
                        print(f"Sesion cerrada: {evento.conductor}")
                    session_id_activo = None

            if estado.estado == Estado.ACTIVA:
                landmarks = detectar_landmarks(frame)
                if landmarks is not None:
                    if mostrar_malla:
                        dibujar_malla_debug(frame, landmarks)
                    else:
                        for x, y in landmarks.puntos_ojo_derecho + landmarks.puntos_ojo_izquierdo:
                            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)

                    ear_derecho = calcular_ear(landmarks.puntos_ojo_derecho)
                    ear_izquierdo = calcular_ear(landmarks.puntos_ojo_izquierdo)
                    ear_promedio = (ear_derecho + ear_izquierdo) / 2
                    mar = calcular_mar(landmarks.puntos_boca)
                    estado_somnolencia, eventos_somnolencia = procesar_somnolencia(
                        estado_somnolencia, ear_promedio, mar, timestamp, config_somnolencia
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

                    yaw, pitch = calcular_yaw_pitch(landmarks.matriz_rotacion)
                    gaze_h_derecho, gaze_v_derecho = calcular_gaze_ratio(
                        landmarks.iris_derecho, landmarks.puntos_ojo_derecho
                    )
                    gaze_h_izquierdo, gaze_v_izquierdo = calcular_gaze_ratio(
                        landmarks.iris_izquierdo, landmarks.puntos_ojo_izquierdo
                    )
                    gaze_horizontal = (gaze_h_derecho + gaze_h_izquierdo) / 2
                    gaze_vertical = (gaze_v_derecho + gaze_v_izquierdo) / 2
                    # Con los ojos cerrados los landmarks de iris no son
                    # confiables; se lo indicamos al detector de distraccion
                    # para que no evalue la senal de mirada sobre esos datos.
                    ojos_abiertos = ear_promedio >= config_somnolencia.ear_umbral
                    estado_distraccion, eventos_distraccion = procesar_pose_y_mirada(
                        estado_distraccion,
                        yaw,
                        pitch,
                        gaze_horizontal,
                        gaze_vertical,
                        ojos_abiertos,
                        timestamp,
                        config_distraccion,
                    )
                    for evento_distraccion in eventos_distraccion:
                        ahora_iso = datetime.now(timezone.utc).isoformat()
                        print(
                            f"Evento de distraccion: {evento_distraccion.tipo} "
                            f"(valor={evento_distraccion.valor:.3f})"
                        )
                        if session_id_activo is not None:
                            registrar_evento(
                                conn,
                                session_id_activo,
                                evento_distraccion.tipo,
                                evento_distraccion.valor,
                                ahora_iso,
                            )
                        else:
                            print("Advertencia: evento de distraccion no persistido, no hay sesion activa en la base de datos.")

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
    parser = argparse.ArgumentParser(
        description="Deteccion de somnolencia y distraccion del conductor."
    )
    parser.add_argument(
        "--malla",
        action="store_true",
        help=(
            "Dibuja la malla facial completa (478 puntos + lineas de "
            "teselacion) y resalta los puntos de control de deteccion "
            "(ojos, iris, boca), en vez de la superposicion normal."
        ),
    )
    args = parser.parse_args()
    main(mostrar_malla=args.malla)
