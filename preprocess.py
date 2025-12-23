
import cv2
def preprocess(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5,5), 0)
