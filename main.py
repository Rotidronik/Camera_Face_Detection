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
# hangszoro inicializalas


def speak_logic(text):
    """Ez a függvény végzi a tényleges beszédet (blokkoló)"""
    try:
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


def speak(text, wait=False):
    """Ez a függvény indítja el a beszédet egy külön háttérszálon (nem blokkoló)
    text: a mondandó
    wait: ha True, megvárja a végét (fagy a kép), ha False, háttérben fut (folyamatos kép)
    """
    def run_speech():
        """Ez a függvény végzi a tényleges beszédet (blokkoló)"""
        try:
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
    if wait:
        run_speech()
    else:
        threading.Thread(target=run_speech).start()  # Háttérben fut


def get_voice_input():
    recogniser = sr.Recognizer()
    with sr.Microphone() as source:

        recogniser.adjust_for_ambient_noise(source, duration=0.1)
        print("Figyelek... Mondd a neved!")
        try:
            audio = recogniser.listen(source, timeout=3, phrase_time_limit=3)
            text = recogniser.recognize_google(audio, language='en-EN')
            return text
        except sr.UnknownValueError:
            print("Sajnos nem értettem, mit mondtál.")
            return None
        except sr.RequestError:
            print("Hálózati hiba a Google szolgáltatásával.")
            return None
        except Exception as e:
            print(f"Hiba történt: {e}")
            return None


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
            print(f"Distance from {name}: {similarity:.4f}")
            if similarity > 0.5:  # ez a kuszobertek h menyi hibat engedunk tavolsagban
                current_name = name
                break
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
    elif key == ord('t'):
        if len(faces) > 0:
            speak(text="Tell your name into the microfone", wait=True)

            name = get_voice_input()

            if name:
                speak(text=f"Welcome to the system, {name}!", wait=False)
                new_face = faces[0].embedding
                new_face_norm = new_face/np.linalg.norm(new_face)
                # uj arc hozzaadasa listaba
                known_faces.append((name, new_face_norm))
                # arc lemetese fileba
                with open(DATA_FILE, "wb") as f:
                    pickle.dump(known_faces, f)
                print("Arc elmentve a fileba: {name}")
            else:
                speak(text="Sorry i didn't hear you clearly.", wait=False)
cam.close()
cv2.destroyAllWindows()
