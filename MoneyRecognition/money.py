import numpy as np
import cv2
import pytesseract
from pytesseract import Output

def process (text):
    text.strip(" \n")
video_capture = cv2.VideoCapture(0)
text = ""
while  process(text) != " ":
    ret, frame = video_capture.read()
    custom_config = r'-l eng --oem 3 --psm 6'
    d = pytesseract.image_to_data(frame, output_type=Output.DICT)
    print(d.keys())
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(text)
video_capture.release()
cv2.destroyAllWindows()