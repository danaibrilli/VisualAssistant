import numpy as np
import cv2
import pytesseract
from pytesseract import Output

def process (text):
    text.strip(" \n")

text = ""

img = cv2.imread("./10euro.jpg")

custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(img)

#text = process(text)
print(text)