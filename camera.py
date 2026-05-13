import cv2


class CamerStream:
    def __init__(self, source=0):
        # Ez fut le az objectum letrehotzasakor
        self.capture = cv2.VideoCapture(source)

    def get_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            return None
        return frame

    def __del__(self):
        self.capture.release()
