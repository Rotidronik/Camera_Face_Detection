import cv2
import numpy as np
import time

# Saját modulok importálása
from camera import CameraStream
import audio_engine
import vision_engine

# Rendszerek indítása
face_app = vision_engine.init_face_analysis()
known_faces = vision_engine.load_known_faces()
cam = CameraStream()
# Háttérben futó hallgatás indítása
audio_engine.start_audio_thread()
audio_engine.speak(text="System online", wait=False)
# ----------------------------------------------------------------------------------------------------

while True:

    frame = cam.get_frame()
    if frame is None:
        break

    faces = face_app.get(frame)
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

        if current_name == "ismeretlen" and audio_engine.trigger_training_name is not None:
            name_to_save = audio_engine.trigger_training_name

            audio_engine.speak(
                text=f"Welcome to the system {name_to_save}!", wait=False)
            vision_engine.save_face(known_faces, name_to_save, embed_norm)

            current_name = name_to_save
            audio_engine.trigger_training_name = None  # Reset

        # ha felismerjuk de megis mondott nevet
        if current_name != "ismeretlen" and audio_engine.trigger_training_name is not None:
            audio_engine.trigger_training_name = None
        # keret rajzolas
        bbox = face.bbox.astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, current_name, (bbox[0], bbox[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # UI megjelenítés
    cv2.imshow("face recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cam.close()
cv2.destroyAllWindows()
