import cv2
import numpy as np
import threading
import multiprocessing
from camera import CameraStream
import audio_engine
import vision_engine
import LLM_engine


trigger_parsed_json = None
llm_is_processing = False


def async_llm_worker(user_name, command_text):
    global trigger_parsed_json, llm_is_processing
    try:
        llm_is_processing = True
        print(f"[LLM SZÁL] Ollama hívás indítva: '{command_text}'...")
        parsed_json = LLM_engine.parse_command(user_name, command_text)
        print(f"[LLM SZÁL] Ollama végzett! Kapott JSON: {parsed_json}")
        trigger_parsed_json = parsed_json
    except Exception as e:
        print(f"Hiba a háttér LLM szálon: {e}")
    finally:
        llm_is_processing = False


def control_smart_home(json_data):
    device = json_data.get("device")
    action = json_data.get("action")
    user = json_data.get("user")

    if device == "none" or action == "unknown":
        print(f"[RENDSZER] Ismeretlen parancs, nincs művelet.")
        return

    if device in ["nagykapu", "kapu"]:
        if action == "open":
            print(
                f"[HARDVER JEL] ---> RELÉ_1 (KAPU) -> HIGH | Felhasználó: {user}")
        elif action == "close":
            print(
                f"[HARDVER JEL] ---> RELÉ_1 (KAPU) -> LOW | Felhasználó: {user}")
    elif device == "redőny":
        if action == "close":
            print(
                f"[HARDVER JEL] ---> MOTOR_2 (REDŐNY) -> DOWN | Felhasználó: {user}")
        elif action == "open":
            print(
                f"[HARDVER JEL] ---> MOTOR_2 (REDŐNY) -> UP | Felhasználó: {user}")
    elif device == "lámpa":
        if action == "on":
            print(
                f"[HARDVER JEL] ---> RELÉ_3 (LÁMPA) -> ON | Felhasználó: {user}")
        elif action == "off":
            print(
                f"[HARDVER JEL] ---> RELÉ_3 (LÁMPA) -> OFF | Felhasználó: {user}")


# ============================================================
# MULTIPROCESSING - Windows alatt kötelező ez a guard!
# ============================================================
if __name__ == "__main__":

    # Rendszerek indítása
    face_app = vision_engine.init_face_analysis()
    known_faces = vision_engine.load_known_faces()

    # Automatikus arc betanítás a 'faces' mappából
    known_faces = vision_engine.auto_register_from_folder(
        face_app, known_faces)
    vision_engine.start_faces_watcher(face_app, known_faces)

    cam = CameraStream()

    # Audio külön processben indul - visszaad egy Queue-t
    command_queue = audio_engine.start_audio_process()

    print("[RENDSZER] Zeusz online. Ismert arcokat várok a kamera előtt...")

    frame_counter = 0
    faces = []

    # ============================================================
    # FŐCIKLUS
    # ============================================================
    while True:
        frame = cam.get_frame()
        if frame is None:
            continue

        # Arcfelismerés csak minden 5. képkockán
        frame_counter += 1
        if frame_counter % 5 == 0:
            faces = face_app.get(frame)

        current_name = "ismeretlen"
        latest_embed_norm = None

        if len(faces) > 0:
            face = faces[0]
            embedding = face.embedding
            latest_embed_norm = embedding / np.linalg.norm(embedding)

            for name, known_embedding in known_faces:
                similarity = np.dot(latest_embed_norm, known_embedding)
                if similarity > 0.5:
                    current_name = name
                    break

            # =========================================================
            # Parancs olvasása a Queue-ból (nem blokkol!)
            # =========================================================
            raw_command = None
            if not command_queue.empty():
                raw_command = command_queue.get_nowait()

            # =========================================================
            # ISMERT ARC: parancs végrehajtása
            # =========================================================
            if current_name != "ismeretlen":
                if raw_command is not None and not llm_is_processing:
                    threading.Thread(
                        target=async_llm_worker,
                        args=(current_name, raw_command),
                        daemon=True).start()

                if trigger_parsed_json is not None:
                    parsed_json = trigger_parsed_json
                    trigger_parsed_json = None
                    control_smart_home(parsed_json)

            # =========================================================
            # ISMERETLEN ARC: parancsot eldobjuk
            # =========================================================
            else:
                if raw_command is not None:
                    print(
                        "[RENDSZER] Ismeretlen arc adott parancsot, figyelmen kívül hagyva.")

            # =========================================================
            # GRAFIKA
            # =========================================================
            bbox = face.bbox.astype(int)
            color = (0, 255, 0) if current_name != "ismeretlen" else (0, 0, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, current_name, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            status = "Zeusz gondolkodik..." if llm_is_processing else "Zeusz figyel"
            status_color = (0, 0, 255) if llm_is_processing else (0, 255, 0)
            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        cv2.imshow("Zeusz - Kapuvezérlő", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cam.close()
    cv2.destroyAllWindows()
