import speech_recognition as sr
import pyttsx3
import numpy as np
import cv2
import face_recognition
import sys, os, subprocess, picamera
import chainer
import argparse
import math
import json
from chainer import cuda
import chainer.functions as F
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
from chainer import serializers
import pytesseract
sys.path.append('./chainer-caption/code')
from CaptionGenerator import CaptionGenerator
import requests
from requests.structures import CaseInsensitiveDict
from collections import defaultdict
import time
import glob


engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[15].id) #1 gia male, 0 gia female
engine.setProperty("rate", 115)

SERVER_IP = '192.168.1.16'


camera = picamera.PiCamera()
camera.resolution = (224, 224)

devnull = open('os.devnull', 'w')

caption_generator=CaptionGenerator(
rnn_model_place='./chainer-caption/data/caption_en_model40.model',
cnn_model_place='./chainer-caption/data/ResNet50.model',
dictonary_place='./chainer-caption/data/MSCOCO/mscoco_caption_train2014_processed_dic.json',
beamsize=5,
depth_limit=30,
gpu_id=-1,
first_word= "<sos>",
)

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


#-------------- COMMAND TO STRING -------------------
def takeCommand(verbal=True):
     
    r = sr.Recognizer()
     
    with sr.Microphone() as source:
         
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)
  
    try:
        print("Recognizing...")   
        query = r.recognize_google(audio, language ='en-in')
        if verbal==True:
            print(f"User said: {query}\n")
  
    except Exception as e:
        print(e)   
        if verbal==True:
            print("Unable to Recognize your voice.") 
        return "None"
     
    return query


#-------------- READ ENCODINGS -------------------
def read_encodings(jfile):
    try:
        f = open(jfile,)
        data = json.load(f)
        #data = json.loads(data)
        f.close()
        return data
    except Exception as e:
        print(e)   
        print("No known encodings.") 
        return "None"

#-------------- TEXT CAPTIONING -------------------
def read(vocal=True):
    text = ""
    camera.capture('image.jpg')

    img = cv2.imread("image.jpg")

    custom_config = r'--oem 1 --psm 10'
    text = pytesseract.image_to_string(img)
    #final_text = re.sub(r'\W+', '', text)
    if text.strip()!='':
        print(text)
        if vocal: speak (text)
    else: speak('No text detected')


#-------------- IMAGE CAPTIONING -------------------
def image_captioning(vocal=True):
    camera.capture('image.jpg')
    captions = caption_generator.generate('image.jpg')
    word = " ".join(captions[0]["sentence"][1:-1])
    print(word)
    if vocal: speak(word)

#-------------- IMGUR UPLOAD -------------------
def imgur_upload():
    headers = {
        'Referer': 'https://imgur.com/upload',
    }

    files = {
        'Filedata': ('"/home/pi/airis/image.jpg";filename', open('/home/pi/airis/image.jpg', 'rb')),
    }

    response = requests.post('https://imgur.com/upload', headers=headers, files=files)
    if response.ok:
        print("img uploaded")
        return 'https://i.imgur.com/'+response.json()['data']['hash']+'.jpg'
    else: return -1


#-------------- OBJECT DETECTION -------------------
def object_detection(img_link,vocal=True): 
    
    url = "http://"+ SERVER_IP+":1912/api/deepdetect/predict"

    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/json"
    #img_link = 'https://i.imgur.com/'+'TfkUELw'+'.jpg'

    data = """
    {
    "service": "detection_600",
    "parameters": {
    "input": {},
    "output": {
    "confidence_threshold": 0.3,
    "bbox": true
    },
    "mllib": {
    "gpu": true
    }
    },
    "data": [
    \""""+img_link+"""\"
    ]
    }

    """

    resp = requests.post(url, headers=headers, data=data)
    resp_json = resp.json()

    classes = resp_json['body']['predictions'][0]['classes']

    count = defaultdict(lambda:0)

    for item in classes:
        count[item['cat']] +=1
    if len(count):
        st = ''
        for item,num in count.items():
            st += str(num) + ' '+ item +', '
        speak(st)
        print(st)
    else: speak("No objects detected")


#-------------- FACE RECOGNITION -------------------
def who(vocal):
    encoding_dict = read_encodings("encodings.json")
    known_face_encodings = []
    known_face_names = []

    if encoding_dict != None:
        for key, value in encoding_dict.items():
            known_face_encodings.append(np.array(value))
            known_face_names.append(key)
    ctr = 0 
    face_names = []
    while True:
        camera.capture('image.jpg')
        frame = cv2.imread("image.jpg")
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        print(face_locations)
        if face_locations != [] : 
            
            for face,encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, encoding)
                name = 'Unknown'
                face_distances = face_recognition.face_distance(known_face_encodings, encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

            break 
        elif ctr >=5:
            break
        else:
            ctr +=1
            continue

    print(face_names)
    n = str(len(face_names)) if len(face_names) else '0'
    num = "I recognize " + n
    print(num)
    n_mid = " person." if len (face_names)==1 else " people."
    num = num + n_mid
    print(num)
    speak(num)
    for face in face_names: speak(face)
    if face_names.count("Unknown")>=1:
        speak ("Do you know the Unknown?")
        answer = takeCommand()
        if 'yes' in answer:
            add_person()

def add_person():
    speak("Tell the unknown person to stand in front of you.")
    face_encodings = []
    while len(face_encodings)!=1:
        camera.capture('image.jpg')
        frame = cv2.imread("image.jpg")
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
    print(face_encodings)
    speak("What is the person's name?")
    name = takeCommand(verbal=False)
    face_encodings = list(face_encodings[0])
    new_data={name:face_encodings}
    #new_json = json.dumps(new_data)
    
    with open ("encodings.json",'r+') as file:
        file_data = json.load(file)
        print(file_data)
        #print(new_json)
        file_data.update(new_data)
        file.seek(0)
        json.dump(file_data, file)


def money(vocal):
    url = "http://"+ SERVER_IP+":1912/api/deepdetect/predict"

    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/json"
    #img_link = 'https://i.imgur.com/'+'TfkUELw'+'.jpg'

    data = """
    {
    "service": "money_model",
    "parameters": {
    "input": {},
    "output": {
    "confidence_threshold": 0.3,
    "bbox": true
    },
    "mllib": {
    "gpu": true
    }
    },
    "data": [
    \""""+img_link+"""\"
    ]
    }

    """

    resp = requests.post(url, headers=headers, data=data)
    resp_json = resp.json()
    
    if resp_json:
        words = "You have "+ resp_json + " euros."
    else: words = "No money detected"
    speak(words)


def barcode(vocal):
    url = "http://"+ SERVER_IP+":1912/api/deepdetect/predict"

    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/json"
    #img_link = 'https://i.imgur.com/'+'TfkUELw'+'.jpg'

    data = """
    {
    "service": "barcode_scanner",
    "parameters": {
    "input": {},
    "output": {
    "confidence_threshold": 0.3,
    "bbox": true
    },
    "mllib": {
    "gpu": true
    }
    },
    "data": [
    \""""+img_link+"""\"
    ]
    }

    """

    resp = requests.post(url, headers=headers, data=data)
    resp_json = resp.json()

    if resp_json:
        words = "This barcode belongs to " + resp_json.
    else: words = "No barcode detected"
    speak(words)


if __name__ == '__main__':
    #clear = lambda: os.system('clear') #for linux
    #clear = lambda: os.system('cls') #for windows
     
    #clear()
    vocal= True
    #if vocal: usrname()

    while True:
        speak("What can I do for you?")
        if vocal: query = takeCommand()
        else: query= input()
        #query= input()
        print(query)

#-------------- IMAGE CAPTIONING -------------------
        if "describe" in query:
            image_captioning(vocal)


#-------------- TEXT RECOGNITION -------------------
        elif ("read" in query and "text" in query) or  ("read" in query and "document" in query):
            read(vocal)


#-------------- OBJECT RECOGNITION -------------------
        elif "objects" in query or "subjects" in query:
            print("i'm in object")
            camera.capture('image.jpg')
            object_detection(imgur_upload(),vocal)


#-------------- FACE RECOGNITION -------------------
        elif "who" in query or "face" in query or "id" in query or "people" in query or "person" in query:
            who(vocal)
        

#-------------- TAKE NOTE -------------------
        elif "take" in query and "note" in query:
            speak("Tell me your note")
            note = takeCommand()
            speak('Do you want to hear your note?')
            if "yes" in takeCommand(): speak(note)
            speak('Give a title to your note')
            title = takeCommand()
            with open (title +'.txt','w') as note_txt:
                note_txt.write(note)
            speak('note saved')

        elif "read" in query and "note" in query:
            speak('Tell me the title of your note')
            title = takeCommand()
            for filename in glob.glob('*.txt'):
                if filename[:-4] == title:
                    with open(title +'.txt','r') as f:
                        note_txt = f.readlines()
                        print(note_txt)
                        speak(note_txt)
            
#-------------- MONEY RECOGNITION -------------------
        elif "money" in query or "count" in query:
            money(vocal)


#-------------- BARCODE RECOGNITION -------------------
        elif "barcode" in query:
            barcode(vocal)

        elif "stop" in query or "exit" in query or "quit" in query or "shut down" in query:
                exit() 
        else: print("no module detected")
        time.sleep(5)