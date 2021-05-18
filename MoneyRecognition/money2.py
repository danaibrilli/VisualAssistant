import numpy as np
import cv2
import re
import pytesseract
import matplotlib.pyplot as plt
from pytesseract import Output

def process (text):
    text.strip(" \n")
    text = re.sub("[^0-9]", "", text)
    return text
text = ""

img = cv2.imread("./MoneyRecognition/50euro_crop.png", cv2.IMREAD_GRAYSCALE)


custom_config = r'--oem 3 --psm 10'
text = pytesseract.image_to_string(img, config=custom_config)

text = process(text)
print(text)