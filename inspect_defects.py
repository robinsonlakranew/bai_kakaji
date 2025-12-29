
import cv2
from preprocess import preprocess
from contour import extract_cap
from config import PLC_DEFECT_BITS, CFG
import defects
import numpy as np 

def load_reference(path):
    d = np.load(path, allow_pickle=True)
    return {
        "version": str(d["version"]),
        "image_shape": tuple(d["image_shape"]),
        "area": float(d["area"]),
        "hull_area": float(d["hull_area"]),
        "sectors": {
            "num_sectors": int(d["num_sectors"]),
            "r_max": d["r_max"],
        },
        "erosion_px": int(d["erosion_px"]),
    }

def load_image(path: str, resize_width: int = None) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    if resize_width is not None:
        h, w = img.shape[:2]
        scale = resize_width / float(w)
        img = cv2.resize(img, (resize_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

def inspect_cap(img_bgr, early_exit=False):
    gray = preprocess(img_bgr)
    contour, mask = extract_cap(gray)
    REF = load_reference("cap_reference.npz")

    if contour is None:
        return {
            "status": "reject",
            "plc_word": 0xFFFF,
            "results": {},
        }

    plc_word = 0
    results = {}

    checks=[
        ("circularity", lambda d: defects.circularity(contour,d)),
        ("flash", lambda d: defects.flash(contour,d)),
        ("short_fill", lambda d: defects.short_fill(contour,gray,REF)),
        ("pinholes", lambda d: defects.pinholes(gray,mask,d)),
        ("blobs", lambda d: defects.blobs(gray,mask,d)),
        ("color", lambda d: defects.color_contamination(img_bgr, mask, d, 'blue')),
    ]

    for name,fn in checks:
        dbg=img_bgr.copy()
        r=fn(dbg)
        results[name]=r
        if not r["pass"]:
            print(name)
            # plc |= (1<<PLC_DEFECT_BITS[name])
            if early_exit: break

    # return {"status":"fail" if plc else "pass","plc_word":plc,"results":results}
    return {
        "status": "fail" if plc_word else "pass",
        "plc_word": plc_word,
        "results": results,
    }