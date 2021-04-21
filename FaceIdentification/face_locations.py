import numpy as np
import cv2
import face_recognition
import matplotlib.pyplot as plt
import matplotlib
import json

def initializing():
    # Load a second sample picture and learn how to recognize it.
    danai = face_recognition.load_image_file("WIN_20210421_13_12_19_Pro.jpg")
    #img = cv2.imread('danai.jpg')
    # Convert from BGR to RGB
    danai_encoding = face_recognition.face_encodings(danai)[0]

    d = {}
    d['Anny'] = danai_encoding.tolist()
    with open('encodings.json', 'w') as fp:
        json.dump(d, fp)
def read_encodings(jfile):
    f = open(jfile,)
    data = json.load(f)
    f.close()
    return data

encoding_dict = read_encodings("encodings.json")
known_face_encodings = []
known_face_names = []

for key, value in encoding_dict.items():
    known_face_encodings.append(np.array(value))
    known_face_names.append(key)

video_capture = cv2.VideoCapture(0)
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    if face_locations != [] : 
        face_names = []
        for face,encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, encoding)
            name = 'Unknown'
            face_distances = face_recognition.face_distance(known_face_encodings, encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
        for face,name in zip(face_locations,face_names):
            cv2.rectangle(frame, (face[3], face[0]), (face[1], face[2]), (255,0,0), 2)
            cv2.rectangle(frame, (face[3], face[2] - 35), (face[1], face[2]), (0, 0, 255))
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (face[3] + 6, face[2] - 6), font, 1.0, (255, 255, 255), 1)
   
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()