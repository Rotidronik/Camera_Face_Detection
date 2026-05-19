import os
import pickle
import numpy as np
from insightface.app import FaceAnalysis

DATA_FILE = "known_faces.pkl"


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
        return []  # known_faces = []


def save_face(known_faces, name, embedding):
    """Elmenti az új arcot a memóriába és a fájlba is."""
    known_faces.append((name, embedding))  # RAM ba menmtjuk
    # mentes a fileba
    with open(DATA_FILE, "wb") as f:
        pickle.dump(known_faces, f)
    print(f"Arc elmentve a fájlba: {name}")
