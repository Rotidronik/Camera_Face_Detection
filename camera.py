import cv2


class CameraStream:
    def __init__(self, source=0):
        # Ez fut le az objectum letrehotzasakor
        self.capture = cv2.VideoCapture(source)

    def get_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            return None
        return frame

    def close(self):
        """Azonnali, parancsra történő hardver-elengedés"""
        if self.capture is not None and self.capture.isOpened():
            self.capture.release()

    def __del__(self):
        self.close()
