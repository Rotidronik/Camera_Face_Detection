import os
import time
import pickle
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

DATA_FILE = "known_faces.pkl"
FACES_FOLDER = "faces"


def init_face_analysis():
    """Inicializálja és előkészíti a neurális hálót."""
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def load_known_faces():
    """Betölti az elmentett arcokat a pkl fájlból."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            known_faces = pickle.load(f)
            print(f"Sikeres betöltés! {len(known_faces)} arcot már ismerek.")
            return known_faces
    else:
        print("Nincs korábbi adatbázis, tiszta lappal indulunk.")
        return []


def save_face(known_faces, name, embedding):
    """Elmenti az új arcot a memóriába és a fájlba is."""
    known_faces.append((name, embedding))
    with open(DATA_FILE, "wb") as f:
        pickle.dump(known_faces, f)
    print(f"Arc elmentve: {name}")


def auto_register_from_folder(face_app, known_faces):
    """
    Automatikusan végignézi a 'faces' mappát induláskor.
    Ha talál új képfájlt (ami még nincs az adatbázisban), betanítja.

    Használat:
        - Hozz létre egy 'faces' mappát a projekt mellett
        - Tegyél bele képeket: robi.jpg, anyu.png stb.
        - A fájlnév (kiterjesztés nélkül) lesz a személy neve
        - Indítsd újra a main.py-t, automatikusan betanítja
    """
    if not os.path.exists(FACES_FOLDER):
        os.makedirs(FACES_FOLDER)
        print(
            f"[REGISZTRÁCIÓ] '{FACES_FOLDER}' mappa létrehozva. Tegyél bele képfájlokat!")
        return known_faces

    known_names = [name for name, _ in known_faces]

    supported = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = [
        f for f in os.listdir(FACES_FOLDER)
        if f.lower().endswith(supported)
    ]

    if not image_files:
        print(
            f"[REGISZTRÁCIÓ] A '{FACES_FOLDER}' mappa üres, nincs mit betanítani.")
        return known_faces

    new_count = 0
    for filename in image_files:
        name = os.path.splitext(filename)[0]

        if name in known_names:
            print(f"[REGISZTRÁCIÓ] '{name}' már ismert, kihagyva.")
            continue

        image_path = os.path.join(FACES_FOLDER, filename)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[REGISZTRÁCIÓ] Nem sikerült betölteni: {filename}")
            continue

        faces = face_app.get(frame)
        if len(faces) == 0:
            print(
                f"[REGISZTRÁCIÓ] Nem találtam arcot ebben a képben: {filename}")
            continue

        if len(faces) > 1:
            print(
                f"[REGISZTRÁCIÓ] Több arc a képen ({filename}), az elsőt használom.")

        embedding = faces[0].embedding
        embedding_norm = embedding / np.linalg.norm(embedding)
        save_face(known_faces, name, embedding_norm)
        new_count += 1
        print(f"[REGISZTRÁCIÓ] ✓ Sikeresen betanítva: '{name}' ({filename})")

    if new_count == 0:
        print(f"[REGISZTRÁCIÓ] Nincs új arc a '{FACES_FOLDER}' mappában.")
    else:
        print(f"[REGISZTRÁCIÓ] {new_count} új arc betanítva!")

    return known_faces


class _FacesFolderHandler(FileSystemEventHandler):
    """Figyeli a 'faces' mappát és ha új kép kerül bele, azonnal betanítja."""

    def __init__(self, face_app, known_faces):
        self.face_app = face_app
        self.known_faces = known_faces

    def on_created(self, event):
        if event.is_directory:
            return

        filepath = event.src_path
        supported = (".jpg", ".jpeg", ".png", ".bmp")
        if not filepath.lower().endswith(supported):
            return

        # Kis várakozás hogy a fájl teljesen megérkezzen (pl. SCP feltöltés)
        time.sleep(1)

        filename = os.path.basename(filepath)
        name = os.path.splitext(filename)[0]
        known_names = [n for n, _ in self.known_faces]

        if name in known_names:
            print(f"[WATCHER] '{name}' már ismert, kihagyva.")
            return

        frame = cv2.imread(filepath)
        if frame is None:
            print(f"[WATCHER] Nem sikerült betölteni: {filename}")
            return

        faces = self.face_app.get(frame)
        if len(faces) == 0:
            print(f"[WATCHER] Nem találtam arcot: {filename}")
            return

        if len(faces) > 1:
            print(
                f"[WATCHER] Több arc a képen ({filename}), az elsőt használom.")

        embedding = faces[0].embedding
        embedding_norm = embedding / np.linalg.norm(embedding)
        save_face(self.known_faces, name, embedding_norm)
        print(f"[WATCHER] ✓ Új arc betanítva újraindítás nélkül: '{name}'")


def start_faces_watcher(face_app, known_faces):
    """Elindítja a mappa figyelőt háttérben."""
    if not os.path.exists(FACES_FOLDER):
        os.makedirs(FACES_FOLDER)

    handler = _FacesFolderHandler(face_app, known_faces)
    observer = Observer()
    observer.schedule(handler, FACES_FOLDER, recursive=False)
    observer.daemon = True
    observer.start()
    print(f"[WATCHER] '{FACES_FOLDER}' mappa figyelése elindult.")
    return observer
