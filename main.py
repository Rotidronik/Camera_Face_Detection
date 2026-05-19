import cv2
import numpy as np
import time
import threading
# Saját modulok importálása
from camera import CameraStream
import audio_engine
import vision_engine
import LLM_engine


trigger_parsed_json = None
llm_is_processing = False
pending_registration_name = None


def async_llm_worker(user_name, command_text):
    """Ez a függvény egy külön szálon fut futásidőben, meghívja az Ollamát,

    és ha megvan a JSON, visszaszól a főszálnak.
    """
    global trigger_parsed_json, llm_is_processing
    try:
        llm_is_processing = True
        print(f"[LLM SZÁL] Ollama hívás indítva: '{command_text}'...")

        parsed_json = LLM_engine.parse_command(user_name, command_text)

        print(f"[LLM SZÁL] Ollama végzett! Kapott JSON: {parsed_json}")
        trigger_parsed_json = parsed_json  # Átadjuk a kész adatot a főszálnak
    except Exception as e:
        print(f"Hiba a háttér LLM szálon: {e}")
    finally:
        llm_is_processing = False

# Hardveres műveletek szimulációja (Ide jönnek majd a tényleges okosotthon relé parancsok)


def control_smart_home(json_data):
    device = json_data.get("device")
    action = json_data.get("action")
    user = json_data.get("user")

    if device == "none" or action == "unknown":
        audio_engine.speak(
            "Sajnálom, nem találtam ilyen eszközt a rendszerben.")
        return

    # Parancsok lekezelése
    if device in ["nagykapu", "kapu"]:
        if action == "open":
            audio_engine.speak(f"Rendben {user}, nyitom a kaput.")
            print("[HARDVER JEL] ---> RELÉ_1 (KAPU) -> HIGH (Nyitás indítva)")
        elif action == "close":
            audio_engine.speak(f"Rendben {user}, zárom a kaput.")
            print("[HARDVER JEL] ---> RELÉ_1 (KAPU) -> LOW (Zárás indítva)")

    elif device == "redőny":
        if action == "close":
            audio_engine.speak("Megértettem, redőnyök leengedése folyamatban.")
            print("[HARDVER JEL] ---> MOTOR_2 (REDŐNY) -> DOWN")
        elif action == "open":
            audio_engine.speak("Megértettem, redőnyök felhúzása folyamatban.")
            print("[HARDVER JEL] ---> MOTOR_2 (REDŐNY) -> UP")


# Rendszerek indítása
face_app = vision_engine.init_face_analysis()
known_faces = vision_engine.load_known_faces()
cam = CameraStream()
# Háttérben futó hallgatás indítása
audio_engine.start_audio_thread()
audio_engine.speak(text="A Rendszer online", wait=False)
# ----------------------------------------------------------------------------------------------------
while True:

    frame = cam.get_frame()
    if frame is None:
        break

    faces = face_app.get(frame)
    current_embedding = None
    current_name = "ismeretlen"
    latest_embed_norm = None

    if len(faces) > 0:
        face = faces[0]
        embedding = face.embedding
        latest_embed_norm = embedding/np.linalg.norm(embedding)

        for name, known_embedding in known_faces:
            # tavolsag az aktualis es imsert arc kozott
            similarity = np.dot(latest_embed_norm, known_embedding)
            # print(f"Distance from {name}: {similarity:.4f}")
            if similarity > 0.5:  # ez a kuszobertek h menyi hibat engedunk tavolsagban
                current_name = name
                break

        # =========================================================
        # 1. ÁG: HA AZ ARC ISMERETLEN (regisztráció)
        # =========================================================
        if current_name == "ismeretlen":

            # Lépés A: Bemondja a nevét
            if audio_engine.trigger_training_name is not None:
                pending_registration_name = audio_engine.trigger_training_name
                audio_engine.trigger_training_name = None
                audio_engine.trigger_llm_command = None  # Biztonsági takarítás

                audio_engine.speak(
                    text=f"Új regisztráció indítva ehhez a névhez: {pending_registration_name}. Kérlek mondd el a biztonsági kódot!",
                    wait=False)

            # Lépés B: Megérkezik a jelszó - KÖZVETLEN PYTHON ELLENŐRZÉS, nem LLM!
            elif audio_engine.trigger_llm_command is not None:
                raw_text = audio_engine.trigger_llm_command
                audio_engine.trigger_llm_command = None

                if pending_registration_name is not None:
                    print(f"raw text: {raw_text}")
                    if "barack" in raw_text.lower():
                        # Helyes jelszó -> regisztráció
                        audio_engine.speak(
                            text=f"Sikeres azonosítás! Üdvözöllek a rendszerben, {pending_registration_name}!",
                            wait=False)
                        vision_engine.save_face(
                            known_faces, pending_registration_name, latest_embed_norm)
                        current_name = pending_registration_name
                        pending_registration_name = None
                    else:
                        # Rossz jelszó -> elutasítás
                        audio_engine.speak(
                            text="Hibás biztonsági kód. Regisztráció elutasítva.",
                            wait=False)
                        pending_registration_name = None
                else:
                    # Nincs regisztráció folyamatban, ismeretlen próbálkozik parancsot adni
                    audio_engine.speak(
                        "Sajnálom, ismeretlen személyeknek nem hajtok végre parancsot.",
                        wait=False)

        # =========================================================
        # 2. ÁG: HA AZ ARC ISMERT (okosotthon parancsok)
        # =========================================================
        else:
            # Biztonsági takarítás: ha ismertként mondaná a nevét, eldobjuk
            if audio_engine.trigger_training_name is not None:
                audio_engine.trigger_training_name = None

            # Lépés A: Parancs érkezik -> LLM feldolgozás
            if audio_engine.trigger_llm_command is not None:
                raw_command = audio_engine.trigger_llm_command
                audio_engine.trigger_llm_command = None

                if not llm_is_processing:
                    threading.Thread(
                        target=async_llm_worker,
                        args=(current_name, raw_command),
                        daemon=True).start()

            # Lépés B: LLM végzett -> parancs végrehajtása
            if trigger_parsed_json is not None:
                parsed_json = trigger_parsed_json
                trigger_parsed_json = None
                control_smart_home(parsed_json)

        # =========================================================
        # GRAFIKA RAJZOLÁSA
        # =========================================================

        # keret rajzolas
        bbox = face.bbox.astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, current_name, (bbox[0], bbox[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # --- ÁLLAPOTJELZŐ SZÖVEGEK A KÉPERNYŐRE ---
        if llm_is_processing:
            cv2.putText(frame, "Zeusz gondolkodik...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif audio_engine.is_speaking:
            cv2.putText(frame, "Zeusz beszel...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            cv2.putText(frame, "Zeusz figyel (Szabad a palya)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # UI megjelenítés
    cv2.imshow("face recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cam.close()
cv2.destroyAllWindows()
