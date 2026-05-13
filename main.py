from camera import CamerStream
import cv2
import keyboard

cam = CamerStream()
while True:
    frame = cam.get_frame()
    if frame is None:
        break

    cv2.imshow("Webcam", frame)
    if keyboard.is_pressed('q'):
        break
