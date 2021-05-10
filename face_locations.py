import numpy as np
import cv2
import face_recognition
import matplotlib.pyplot as plt
import matplotlib

# Load a second sample picture and learn how to recognize it.
hiddleston_image = face_recognition.load_image_file("tom_hiddleston.jpg")
hiddleston_face_encoding = face_recognition.face_encodings(hiddleston_image)[0]
print(hiddleston_face_encoding)


video_capture = cv2.VideoCapture(0)
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    faceloc = face_recognition.face_locations(frame)
    if faceloc != [] : 
        for face in faceloc:
            cv2.rectangle(frame, (face[3], face[0]), (face[1], face[2]), (255,0,0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()