import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognizer:
    def __init__(self):
        self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def get_embedding(self, frame):
        faces = self.app.get(frame)
        if len(faces) > 0:
            return faces[0].embedding
        return None

    def compare_faces(self, known_faces, current_embedding, threshold=1.2):
        # A két vektor közti "távolság" (cosine distance)
        for name, known_embedding in known_faces:
            dist = np.sum(np.square(current_embedding - known_embedding))
            if dist < threshold:
                return name
        return "Unknown"
