import cv2
import numpy as np
import time
import requests
import multiprocessing
import threading  # Új import az aszinkron makróhoz
import vision_engine
import LLM_engine
import audio_engine
from camera import CameraStream


# Állapotváltozó, hogy ne lehessen kétszer egyszerre elindítani a makrót
macro_is_running = False


def send_gate_command_to_ha():
    """Nyers Broadlink RF jel kiküldése a Home Assistantnak."""
    try:
        url = f"{HA_URL}/api/services/remote/send_command"
        payload = {
            "entity_id": "remote.broadlink_kapu_nyito",
            "device": "roger_kapu",
            "command": "open"
        }
        response = requests.post(url, json=payload, headers=HEADERS, timeout=2)
        return response.status_code in [200, 201]
    except Exception as e:
        print(f"[HA API TRANSPORT HIBA] {e}")
        return False


def automated_gate_sequence():
    """
    Aszinkron háttérszál, ami levezényli a Nyitás -> Megállítás -> Zárás szekvenciát
    anélkül, hogy a fő arcfelismerő programot leblokkolná.
    """
    global macro_is_running
    macro_is_running = True

    print("\n=======================================================")
    print("[MAKRÓ] ---> AUTOMATIZÁLT KAPU SZEKVENCIA ELINDULT <---")
    print("=======================================================")

    # 1. LÉPÉS: Nyitás indítása
    print("[MAKRÓ] 1/3: Első jel kiküldése -> Kapu NYITÁSA elindult...")
    send_gate_command_to_ha()

    # Várakozás, amíg a kapu kinyílik annyira, hogy kényelmesen be lehessen férni (1.5 másodperc)
    time.sleep(20)

    # 2. LÉPÉS: Kapu megállítása részlegesen nyitott állapotban
    print("[MAKRÓ] 2/3: Második jel kiküldése -> Kapu MEGÁLLÍTÁSA...")
    send_gate_command_to_ha()

    # Áthaladási időablak: Ennyi ideig áll nyitva a kapu, amíg átsétálsz/bemész a kocsival (7 másodperc)
    print("[MAKRÓ] Várakozás az áthaladásra (7 másodperc)...")
    time.sleep(10.0)

    # 3. LÉPÉS: Kapu bezárása
    print("[MAKRÓ] 3/3: Harmadik jel kiküldése -> Kapu ZÁRÁSA...")
    send_gate_command_to_ha()

    print("=======================================================")
    print("[MAKRÓ] ---> AUTOMATIZÁLT KAPU SZEKVENCIA KÉSZ <---")
    print("=======================================================\n")
    macro_is_running = False


# ============================================================
# FŐPROGRAM
# ============================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()

    print("[RENDSZER] Zeusz 4.1 (Aszinkron Makró Verzió) indul...")

    # AI modulok és hardveres kapcsolatok felélesztése
    face_app = vision_engine.init_face_analysis()
    known_faces = vision_engine.load_known_faces()
    known_faces = vision_engine.auto_register_from_folder(
        face_app, known_faces)
    vision_engine.start_faces_watcher(face_app, known_faces)

    cam = CameraStream()
    command_queue = audio_engine.start_audio_process()

    last_authorized_time = 0
    loop_counter = 0

    print("[RENDSZER] Zeusz online. Valós idejű figyelés aktív. Makró parancsra vár...")

    while True:
        frame = cam.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        loop_counter += 1

        # Arcfelismerés futtatása ~0.3 másodpercenként
        if loop_counter % 3 == 0:
            faces = face_app.get(frame)
            if len(faces) > 0:
                face = faces[0]
                embedding = face.embedding
                embedding_norm = embedding / np.linalg.norm(embedding)

                for name, known_embedding in known_faces:
                    similarity = np.dot(embedding_norm, known_embedding)
                    if similarity > 0.5:
                        if time.time() - last_authorized_time > 15:
                            print(
                                f"[BIZTONSÁG] ✓ Arc azonosítva: Szia {name}! Rendszer élesítve...")
                        last_authorized_time = time.time()
                        break

        # Hangparancsok ellenőrzése
        if not command_queue.empty():
            raw_command = command_queue.get_nowait()
            print(
                f"[FŐSZÁL] Feldolgozandó hangparancs érkezett: '{raw_command}'")

            if time.time() - last_authorized_time < 15:
                print("[LLM] Nyelvi modell (Ollama) elemzés...")
                parsed_json = LLM_engine.parse_command("robi", raw_command)

                if parsed_json and parsed_json.get("device") == "kapu":
                    # Megnézzük, hogy nem fut-e már a makró, nehogy egymásra küldje a jeleket
                    if not macro_is_running:
                        print(
                            f"[REAKCIÓ] Kapu szándék észlelve. Makró szál indítása!")
                        # ELINDÍTJUK A SZEKVENCIÁT EGY KÜLÖNÁLLÓ HÁTTÉRSZÁLON
                        threading.Thread(
                            target=automated_gate_sequence, daemon=True).start()
                    else:
                        print(
                            "[RENDSZER] A kapu automatizálási sorozat már folyamatban van, parancs blokkolva.")
            else:
                print(
                    "[BLOKKOLÁS] Elhangzott a parancs, de nincs ismert arc az elmúlt 15 másodpercben.")

        time.sleep(0.1)
