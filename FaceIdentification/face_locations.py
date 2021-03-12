import numpy as np
import cv2
import face_recognition
import matplotlib.pyplot as plt
import matplotlib


video_capture = cv2.VideoCapture(0)
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    faceloc = face_recognition.face_locations(frame)
    if faceloc != [] : 
        faceloc = faceloc[0]
        cv2.rectangle(frame, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (255,0,0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()