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

    if capturadas < FOTOS_POR_CONDUCTOR:
        print(
            f"Enrolamiento incompleto: solo se capturaron {capturadas}/{FOTOS_POR_CONDUCTOR} "
            "fotos (la camara dejo de entregar frames). No se registro en la base de datos."
        )
        sys.exit(1)

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
