import cv2
import threading


class CameraStream:
    def __init__(self):
        rtsp_url = "rtsp://admin:Rotidronik2002@192.168.10.120:554/h264Preview_01_sub"
        print(f"[CAMERA] Kapcsolódás a Reolinkhez: {rtsp_url}")
        self.capture = cv2.VideoCapture(rtsp_url)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.frame = None
        self.lock = threading.Lock()
        self.running = True

        # Háttérszál ami folyamatosan olvassa a kamerát
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        print("[CAMERA] Háttér olvasó szál elindítva.")

    def _reader(self):
        """Folyamatosan olvassa a kamerát háttérben, mindig a legfrissebb képet tartja."""
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                continue
            with self.lock:
                self.frame = frame  # Mindig felülírjuk, régi képkockák eldobva

    def get_frame(self):
        """Azonnal visszaadja a legutolsó képkockát, nem vár a hálózatra."""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def close(self):
        self.running = False
        if self.capture is not None and self.capture.isOpened():
            self.capture.release()
            print("[CAMERA] Kapcsolat lezárva.")

    def __del__(self):
        self.close()
