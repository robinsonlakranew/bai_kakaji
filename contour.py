
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


# def find_main_contour_otsu(gray: np.ndarray):
#     """
#     Simple + safe contour extraction.
#     Always returns the OUTER envelope of the cap.

#     Returns:
#         contour_main (or None),
#         area_px (float),
#         binary_mask_used (uint8 0/255)
#     """
#     # 1. Otsu threshold
#     _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     # 2. Ensure connectivity
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
#     thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)

#     # 3. Find contours on binary
#     contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return None, 0.0, thr

#     # 4. Pick largest region (cap body)
#     c = max(contours, key=cv2.contourArea)
#     area = float(cv2.contourArea(c))

#     # 5. Convert contour → filled mask
#     h, w = gray.shape[:2]
#     mask = np.zeros((h, w), dtype=np.uint8)
#     cv2.drawContours(mask, [c], -1, 255, thickness=-1)

#     # 6. Re-extract contour from filled mask
#     #    THIS guarantees outer envelope
#     contours2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     if not contours2:
#         return None, 0.0, mask

#     contour_main = max(contours2, key=cv2.contourArea)
#     area = float(cv2.contourArea(contour_main))

#     return contour_main, mask

# k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))

# c, mask = find_main_contour_otsu(gray)
# mask = cv2.erode(mask, k)

# (cx, cy), r = cv2.minEnclosingCircle(c)

# cv2.circle(
#     mask,
#     (int(cx), int(cy)),
#     int(r * 0.60),   # 🔑 tighter
#     255,
#     -1
# )


