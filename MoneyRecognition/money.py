import cv2
import pytesseract

video_capture = cv2.VideoCapture(0)
ret, frame = video_capture.read()

print(pytesseract.image_to_string(frame))