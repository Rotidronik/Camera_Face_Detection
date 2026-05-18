from camera import CameraStream
import cv2
import numpy as np
import speech_recognition as sr
from insightface.app import FaceAnalysis
import pyttsx3
import time
import threading
import pickle
import os

DATA_FILE = "known_faces.pkl"

# --- GLOBÁLIS ÁLLAPOTVÁLTOZÓ A SZÁLAK KÖZÖTTI KOMMUNIKÁCIÓHOZ ---
# Ha a háttérszál nevet talál, ide készíti be a főszálnak
trigger_training_name = None
is_speaking = False  # Globális flag
# hangszoro inicializalas


def speak(text, wait=False):
    """Ez a függvény indítja el a beszédet egy külön háttérszálon (nem blokkoló)
    text: a mondandó
    wait: ha True, megvárja a végét (fagy a kép), ha False, háttérben fut (folyamatos kép)
    """
    global is_speaking

    def run_speech():
        """Ez a függvény végzi a tényleges beszédet (blokkoló)"""
        global is_speaking
        try:
            is_speaking = True
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')

            for voice in voices:
                if "Hungarian" in voice.name or "hu-HU" in voice.id:
                    engine.setProperty('voice', voice.id)
                    break
            engine.setProperty('rate', 180)  # beszedsebesség
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine
        except Exception as e:
            print(f"Hiba a beszednel: {e}")
        finally:
            time.sleep(0.2)
            is_speaking = False
    if wait:
        run_speech()
    else:
        threading.Thread(target=run_speech).start()  # Háttérben fut


def continuous_audio_listener():
    """Ez a függvény egy külön háttérszálon fut futásidőben végig.

    Folyamatosan figyel a mikrofonra, és ha meghallja, hogy 'Zeus',
    feldolgozza a nevet.
    """
    global trigger_training_name, is_speaking
    recogniser = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recogniser.adjust_for_ambient_noise(source=source, duration=0.8)

    print("-> Siri ébresztési szó figyelése elindult a háttérben...")
    while True:
        # Ha a főszál éppen ment egy arcot, várunk, nem szakítjuk félbe
        if trigger_training_name is not None or is_speaking:
            time.sleep(0.2)
            continue
        with mic as source:
            try:
                # 4s ig figyel
                audio = recogniser.listen(
                    source=source, timeout=2, phrase_time_limit=4)
                text = recogniser.recognize_google(
                    audio, language="en-US").lower()

                if "siri" in text:
                    # print("[HÁTTÉRSZÁL] Siri aktiválva!")
                    if "my name is" in text:
                        extracted_name = text.split(
                            "my name is")[-1].strip()  # kiszedjuk a nevet
                        extracted_name = (extracted_name.replace(
                            ".", "").replace("?", "").strip())
                        if extracted_name:
                            # print(f"[HÁTTÉRSZÁL] Név bekészítve mentésre: {extracted_name}")
                            trigger_training_name = (extracted_name)

            except (sr.WaitTimeoutError, sr.UnknownValueError):
                # Ha csend van vagy nem érthető a zaj, hibajelzés nélkül megy tovább a háttér ciklus
                continue
            except Exception as e:
                print(f"Beszéd hiba a háttérben: {e}")
                time.sleep(1)
# ----------------------------------------------------------------------------------------------------


# inicializing AI
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

cam = CameraStream()
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "rb") as f:
        known_faces = pickle.load(f)
        print(f"Sikeres betöltés! {len(known_faces)} arcot már ismerek.")
else:
    known_faces = []
    print("Nincs korábbi adatbázis, tiszta lappal indulunk.")

audio_thread = threading.Thread(target=continuous_audio_listener, daemon=True)
audio_thread.start()

speak(text="System online", wait=False)

while True:

    frame = cam.get_frame()
    if frame is None:
        break

    faces = app.get(frame)
    current_embedding = None
    current_name = "ismeretlen"

    if len(faces) > 0:
        face = faces[0]
        embedding = face.embedding
        embed_norm = embedding/np.linalg.norm(embedding)

        for name, known_embedding in known_faces:
            # tavolsag az aktualis es imsert arc kozott
            similarity = np.dot(embed_norm, known_embedding)
            # print(f"Distance from {name}: {similarity:.4f}")
            if similarity > 0.5:  # ez a kuszobertek h menyi hibat engedunk tavolsagban
                current_name = name
                break
        if current_name == "ismeretlen" and trigger_training_name is not None:
            name_to_save = trigger_training_name
            # Üdvözlés szálon (nem akasztja meg a videót)
            speak(text=f"Welcome to the system {name_to_save}!", wait=False)
            known_faces.append((name_to_save, embed_norm))  # RAM ba menmtjuk

            # mentes a fileba
            with open(DATA_FILE, "wb") as f:
                pickle.dump(known_faces, f)
            # print(f"Arc elmentve a fileba: {name_to_save}")
            # RESET: Töröljük a triggert, hogy a következő képkockát ne mentse el újra és újra
            trigger_training_name = None

        # ha felismerjuk de megis mondott nevet
        if current_name != "ismeretlen" and trigger_training_name is not None:
            trigger_training_name = None
        # keret rajzolas
        bbox = face.bbox.astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, current_name, (bbox[0], bbox[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("face recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cam.close()
cv2.destroyAllWindows()
