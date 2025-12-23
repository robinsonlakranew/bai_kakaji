
import cv2
from .preprocess import preprocess
from .contour import extract_cap
from .config import PLC_DEFECT_BITS, CFG
from . import defects

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
    if contour is None:
        return {"status":"reject","plc_word":0xFFFF}

    plc=0
    results={}

    checks=[
        ("circularity", lambda d: defects.circularity(contour,d)),
        ("flash", lambda d: defects.flash(contour,d)),
        ("short_fill", lambda d: defects.short_fill(contour,d)),
        ("pinholes", lambda d: defects.pinholes(gray,mask,d)),
        ("blobs", lambda d: defects.blobs(gray,mask,d)),
        ("chips", lambda d: defects.chips(contour,d)),
        ("cracks", lambda d: defects.cracks(gray,mask,d)),
    ]

    for name,fn in checks:
        dbg=img_bgr.copy()
        r=fn(dbg)
        results[name]=r
        if not r["pass"]:
            plc |= (1<<PLC_DEFECT_BITS[name])
            if early_exit: break

    return {"status":"fail" if plc else "pass","plc_word":plc,"results":results}
