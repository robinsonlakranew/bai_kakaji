
import cv2
import numpy as np

def extract_cap(gray):
    _, thr = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = cv2.morphologyEx(
        thr, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    )
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [c], -1, 255, -1)
    return c, mask
