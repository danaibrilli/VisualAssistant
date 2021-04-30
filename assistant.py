import os
import shutil
import speech_recognition as sr
import pyttsx3 #text to speech
import numpy as np
import cv2
import face_recognition
import json
import re
import pytesseract



engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id) #1 gia male, 0 gia female


def speak(audio):
    engine.say(audio)
    engine.runAndWait()


def usrname():

    try :
        j = read_encodings("user.json")
        uname = j['name']
        speak("Welcome back")
    except Exception as e:
        print("No user file found.") 
        speak("What should i call you")
        uname = takeCommand()
        speak("Welcome")
    
    speak(uname)
    d = {}
    d['name'] = uname
    with open('user.json', 'w') as fp:
        json.dump(d, fp)
    
    speak("How can i Help you")

def takeCommand():
     
    r = sr.Recognizer()
     
    with sr.Microphone() as source:
         
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)
  
    try:
        print("Recognizing...")   
        query = r.recognize_google(audio, language ='en-in')
        print(f"User said: {query}\n")
  
    except Exception as e:
        print(e)   
        print("Unable to Recognize your voice.") 
        return "None"
     
    return query



def read_encodings(jfile):
    try:
        f = open(jfile,)
        data = json.load(f)
        f.close()
        return data
    except Exception as e:
        print(e)   
        print("No known encodings.") 
        return "None"


def add_person():
    speak("Tell the unknown person to stand in front of you.")
    video_capture = cv2.VideoCapture(0)
    face_encodings = []
    while len(face_encodings)!=1:
        ret, frame = video_capture.read()
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
    speak("What is the person's name?")
    name = takeCommand()
    #add name & encodings to json file



def who():
    speak("Opening your camera")
    encoding_dict = read_encodings("./FaceIdentification/encodings.json")
    known_face_encodings = []
    known_face_names = []
    
    if encoding_dict != None:
        for key, value in encoding_dict.items():
            known_face_encodings.append(np.array(value))
            known_face_names.append(key)
    
    video_capture = cv2.VideoCapture(0)
    while True:
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

            break 
        else:
            continue

    print(face_names)
    num = "I recognize" + str(len(face_names)) 
    num = num + " person." if len (face_names)==1 else "people."
    speak(num)
    for face in face_names: speak(face)
    if face_names.count("Unknown")>=1:
        speak ("Do you know the Unknown?")
        answer = takeCommand()
        if 'yes' in answer:
            add_person()
    
    video_capture.release()
    cv2.destroyAllWindows()

def process (text):
    text.strip(" \n")

def read():
    text = ""

    img = cv2.imread("./TextReading/Document.jpg")

    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(img)
    final_text = re.sub(r'\W+', '', text)
    speak (final_text)

if __name__ == '__main__':
    clear = lambda: os.system('clear')
     
    clear()
    usrname()
    query = takeCommand()
    print(query)
    if "Who" in query or "who" in query:
        who()
    if "Read" in query or "read" in query or "text" in query:
        read()