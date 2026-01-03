from defects import ColorContaminationCheck
from preprocess import preprocess
from contour import extract_cap
from config import CFG

import numpy as np
import cv2

def extract_main_contour(gray):
    _, thr = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    return max(contours, key=cv2.contourArea)


def load_image(path: str, resize_width: int = None) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    if resize_width is not None:
        h, w = img.shape[:2]
        scale = resize_width / float(w)
        img = cv2.resize(img, (resize_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

### reference code for color contamination

reference_img = load_image("/content/34_Piece_image_0.bmp", resize_width=CFG['resize_width'])
reference_gray = preprocess(reference_img)
reference_contour_main,reference_mask = extract_cap(reference_gray)

imgs_with_masks = [(reference_img, reference_mask)] #, (img2, mask2)]

checker = ColorContaminationCheck()

# TODO: load your own images + masks here
checker.calibrate_from_examples("blue", imgs_with_masks)


############# reference for cracks 

# reference_img = load_image("/content/10_Piece_image_0.bmp", resize_width=CFG['resize_width'])
# gray = preprocess(reference_img)

import cv2
import numpy as np
from defects import (
    contour_orientation_pca,
    rotate_image_and_contour,
    resolve_180_flip,
    # erode_contour,
    compute_sector_profile,
)

def build_reference_from_image(
    img_bgr,
    gray,
    num_sectors=16,
    erosion_px=3,
):
    # gray = preprocess(img_bgr)

    contour = extract_main_contour(gray)
    if contour is None:
        raise RuntimeError("No contour found in reference image")

    # --- rotation normalization ---
    angle = contour_orientation_pca(contour)
    _, contour = rotate_image_and_contour(gray, contour, angle)
    contour = resolve_180_flip(contour)

    # # --- rim erosion ---
    # if erosion_px > 0:
    #     contour = erode_contour(contour, erosion_px, gray.shape)

    # --- geometry ---
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    # --- sectorization ---
    sectors = compute_sector_profile(contour, num_sectors)

    ref = {
        "version": "1.0",
        "image_shape": gray.shape,
        "area": float(area),
        "hull_area": float(hull_area),
        "sectors": {
            "num_sectors": num_sectors,
            "r_max": sectors["r_max"].astype(np.float32),
        },
        "erosion_px": erosion_px,
    }

    return ref


def save_reference(ref, path):
    np.savez(
        path,
        version=ref["version"],
        image_shape=ref["image_shape"],
        area=ref["area"],
        hull_area=ref["hull_area"],
        num_sectors=ref["sectors"]["num_sectors"],
        r_max=ref["sectors"]["r_max"],
        erosion_px=ref["erosion_px"],
    )

ref = build_reference_from_image(reference_img, reference_gray)
save_reference(ref, "cap_reference.npz")