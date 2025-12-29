
import cv2, numpy as np
from timing import timed
from config import CFG
import math
from pathlib import Path
import json

def compute_circularity_metrics(contour: np.ndarray) -> dict[str, float]:
    """
    Compute shape metrics from outer contour.
    Safe for production: guards all degenerate cases.
    """
    area = float(cv2.contourArea(contour))
    perim = float(cv2.arcLength(contour, True))

    # --- Circularity ---
    circularity = (4.0 * math.pi * area / (perim * perim)) if perim > 1e-8 else 0.0

    # --- Min enclosing circle ---
    (cx, cy), radius = cv2.minEnclosingCircle(contour)

    # --- Centroid ---
    M = cv2.moments(contour)
    if M["m00"] > 1e-8:
        centroid_x = M["m10"] / M["m00"]
        centroid_y = M["m01"] / M["m00"]
    else:
        centroid_x, centroid_y = cx, cy

    # --- Ellipse / axis ratio ---
    axis_ratio = 0.0
    if len(contour) >= 5:
        (_, _), (a, b), _ = cv2.fitEllipse(contour)
        major = max(a, b)
        minor = min(a, b)
        if major > 1e-8:
            axis_ratio = minor / major

    # --- Radial variance ---
    pts = contour.reshape(-1, 2).astype(np.float32)
    dx = pts[:, 0] - centroid_x
    dy = pts[:, 1] - centroid_y
    dists = np.sqrt(dx * dx + dy * dy)

    if dists.size:
        mean_r = float(dists.mean())
        radial_var = float(dists.var() / (mean_r * mean_r + 1e-12))
    else:
        mean_r = 0.0
        radial_var = float("inf")

    return {
        "area": area,
        "perimeter": perim,
        "circularity": circularity,
        "axis_ratio": axis_ratio,
        "mean_radius": mean_r,
        "radial_variance": radial_var,
        "center_x": float(cx),
        "center_y": float(cy),
        "radius": float(radius),
        "centroid_x": float(centroid_x),
        "centroid_y": float(centroid_y),
    }

@timed
def circularity(
    contour: np.ndarray,
    dbg: np.ndarray,
    cfg: dict = CFG
) -> dict[str, any]:

    # ---- Compute metrics ----
    metrics = compute_circularity_metrics(contour)

    circ = metrics["circularity"]
    axis_ratio = metrics["axis_ratio"]
    radial_var = metrics["radial_variance"]

    # ---- Individual checks ----
    circ_ok = circ >= cfg["circularity_tol"]
    axis_ok = axis_ratio >= cfg["axis_ratio_tol"]
    radial_ok = radial_var <= cfg["radial_variance_tol"]

    passed = circ_ok and axis_ok and radial_ok

    # ---- Debug overlay ----
    color = (0, 255, 0) if passed else (0, 0, 255)
    cv2.drawContours(dbg, [contour], -1, color, 2)

    txt1 = f"C:{circ:.3f} AR:{axis_ratio:.3f} RV:{radial_var:.4f}"
    txt2 = "OK" if passed else "CIRC_FAIL"

    cv2.putText(
        dbg, txt1, (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA
    )
    cv2.putText(
        dbg, txt2, (8, 48),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
    )

    return {
        "pass": bool(passed),
        "score": float(circ),  # primary score, like before
        "details": {
            "circularity": circ,
            "axis_ratio": axis_ratio,
            "radial_variance": radial_var,
            "circularity_ok": circ_ok,
            "axis_ratio_ok": axis_ok,
            "radial_var_ok": radial_ok,
        },
        "overlay": dbg
    }

@timed
def flash(contour, dbg):
    """
    CVS3000-style flash detection using profile protrusion,
    with the SAME input/output contract as the original flash().
    """

    # ---------------- CONFIG ----------------
    min_height_px = CFG.get("flash_min_height_px", 11.0)
    min_run_pts   = CFG.get("flash_min_run_pts", 6)
    smooth_ksize  = CFG.get("flash_smooth_ksize", 9)
    # ----------------------------------------

    # --- 1. Center ---
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return {"pass": True, "score": 0.0, "overlay": dbg}

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    # --- 2. Radial profile ---
    pts = contour.reshape(-1, 2).astype(np.float32)
    radii = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)

    # --- 3. Smooth (knurling suppression) ---
    radii_s = cv2.GaussianBlur(
        radii.reshape(-1, 1),
        (smooth_ksize | 1, 1),  # ensure odd
        0
    ).flatten()

    # --- 4. Reference radius ---
    r_ref = np.median(radii_s)

    # --- 5. Protrusion ---
    dev = radii_s - r_ref
    flash_flags = dev > min_height_px

    # --- 6. Consecutive run check ---
    max_run = run = 0
    for f in flash_flags:
        if f:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0

    is_flash = max_run >= min_run_pts

    # --- 7. Score (PLC-friendly, normalized) ---
    rMaxOut = float(dev.max())
    score = rMaxOut / (r_ref + 1e-6)   # similar spirit to hull-area ratio

    # --- 8. Debug overlay ---
    if is_flash:
        for i, f in enumerate(flash_flags):
            if f:
                cv2.circle(
                    dbg,
                    tuple(pts[i].astype(int)),
                    1,
                    (0, 0, 255),
                    -1
                )

    return {
        "pass": not is_flash,
        "score": score,
        "overlay": dbg
    }


def bgr_to_lab(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] = lab[..., 0] * (100.0 / 255.0)
    lab[..., 1] -= 128.0
    lab[..., 2] -= 128.0
    return lab


def lab_mean_std_from_mask(lab_img: np.ndarray, mask: np.ndarray):
    m = mask > 0
    if not m.any():
        return np.zeros(3), np.zeros(3)
    pixels = lab_img[m]
    return pixels.mean(axis=0), pixels.std(axis=0)


def delta_e_cie76(lab1, lab2) -> float:
    return float(np.linalg.norm(np.asarray(lab1) - np.asarray(lab2)))

def mask_center_and_edge(mask, erosion_ratio=0.35):
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return mask_u8*0, mask_u8*0

    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()
    radius = max(1, int(0.5 * min(maxx-minx+1, maxy-miny+1)))

    erosion = max(1, int(radius * erosion_ratio))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (erosion*2+1, erosion*2+1)
    )

    center = cv2.erode(mask_u8, kernel)
    edge = cv2.subtract(mask_u8, center)
    return center, edge

def generate_patch_masks(mask, ph, pw):
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return []

    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()

    patches = []
    for y in range(miny, maxy, ph):
        for x in range(minx, maxx, pw):
            pm = np.zeros_like(mask_u8)
            pm[y:y+ph, x:x+pw] = 255
            pm = cv2.bitwise_and(pm, mask_u8)
            if pm.sum() > 0:
                patches.append(pm)
    return patches

class ColorContaminationCheck:
    def __init__(self, cfg=None):
        self.cfg = dict(DEFAULT_CFG)
        if cfg:
            self.cfg.update(cfg)
        self.references = {}
        self._load_references_on_init()

    def _load_references_on_init(self):
        p = Path(self.cfg["reference_save_path"])
        if p.exists():
            with p.open("r") as f:
                self.references = json.load(f)

    def save_references(self):
        with open(self.cfg["reference_save_path"], "w") as f:
            json.dump(self.references, f, indent=2)

    def calibrate_from_examples(self, color_name, imgs_with_masks):
        lab_means, center_means, edge_means = [], [], []

        for img, mask in imgs_with_masks:
            lab = bgr_to_lab(img)
            mean, _ = lab_mean_std_from_mask(lab, mask)
            lab_means.append(mean)

            cm, em = mask_center_and_edge(mask)
            cmean, _ = lab_mean_std_from_mask(lab, cm)
            emean, _ = lab_mean_std_from_mask(lab, em)
            center_means.append(cmean)
            edge_means.append(emean)

        ref = {
            "mean_LAB": np.mean(lab_means, axis=0).tolist(),
            "center_mean_LAB": np.mean(center_means, axis=0).tolist(),
            "edge_mean_LAB": np.mean(edge_means, axis=0).tolist(),
            "n_examples": len(lab_means),
        }

        self.references[color_name] = ref
        self.save_references()
        return ref
        
    def run(self, img_bgr, mask, color_name, annotate=True):
        lab = bgr_to_lab(img_bgr)
        mean, _ = lab_mean_std_from_mask(lab, mask)

        ref = self.references[color_name]
        delta_global = delta_e_cie76(mean, ref["mean_LAB"])

        center, edge = mask_center_and_edge(mask)
        cm, _ = lab_mean_std_from_mask(lab, center)
        em, _ = lab_mean_std_from_mask(lab, edge)

        delta_center = delta_e_cie76(cm, ref["center_mean_LAB"])
        delta_edge = delta_e_cie76(em, ref["edge_mean_LAB"])

        fail = (
            delta_global > self.cfg["global_delta_e_threshold"] or
            delta_edge > self.cfg["edge_delta_e_threshold"]
        )

        decision = {
            "pass": not fail,
            "delta_global": delta_global,
            "delta_edge": delta_edge
        }

        if not annotate:
            return decision, None

        out = img_bgr.copy()
        cv2.putText(
            out,
            f"ΔE:{delta_global:.2f}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0) if not fail else (0,0,255),
            2
        )
        return decision, out

_color_checker = ColorContaminationCheck({
    "reference_save_path": '/content/color_references.json'
})

@timed
def color_contamination(img_bgr, mask, dbg, color_name=None):
    """
    Color contamination defect check.
    Input matches other defects:
      - uses existing mask
      - writes on dbg
    """

    decision, overlay = _color_checker.run(
        img_bgr=img_bgr,
        mask=mask,
        color_name=color_name or CFG["cap_color"],
        annotate=True
    )
    # print(out)
    # decision = out["decision"]
    # overlay = out["annotated_image"]

    # If annotate produced an overlay, use it
    if overlay is not None:
        dbg[:] = overlay[:]

    return {
        "pass": decision["pass"],
        "score": {
            "delta_global": decision["delta_global"],
            "delta_edge": decision["delta_edge"],
        },
        "overlay": dbg
    }

def contour_orientation_pca(contour):
    pts = contour.reshape(-1, 2).astype(np.float32)
    mean = np.mean(pts, axis=0)
    pts_centered = pts - mean

    _, _, vt = np.linalg.svd(pts_centered, full_matrices=False)
    direction = vt[0]  # principal axis

    angle = np.degrees(np.arctan2(direction[1], direction[0]))
    return angle

def rotate_image_and_contour(img, contour, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    img_rot = cv2.warpAffine(img, M, (w, h))

    contour_rot = cv2.transform(contour, M)
    return img_rot, contour_rot

def resolve_180_flip(contour):
    pts = contour.reshape(-1, 2)
    cx = np.mean(pts[:,0])

    left_area = np.sum(pts[:,0] < cx)
    right_area = np.sum(pts[:,0] >= cx)

    if right_area < left_area:
        # flip 180°
        M = cv2.getRotationMatrix2D((cx, np.mean(pts[:,1])), 180, 1.0)
        contour = cv2.transform(contour, M)

    return contour

def contour_to_polar(contour):
    pts = contour.reshape(-1, 2)
    cx, cy = np.mean(pts, axis=0)

    dx = pts[:,0] - cx
    dy = pts[:,1] - cy

    r = np.sqrt(dx*dx + dy*dy)
    theta = (np.degrees(np.arctan2(dy, dx)) + 360) % 360

    return r, theta, (cx, cy)

def compute_sector_profile(contour, num_sectors=16):
    r, theta, center = contour_to_polar(contour)

    sector_bins = np.linspace(0, 360, num_sectors + 1)
    sector_r_max = np.zeros(num_sectors)

    for i in range(num_sectors):
        mask = (theta >= sector_bins[i]) & (theta < sector_bins[i+1])
        if np.any(mask):
            sector_r_max[i] = np.max(r[mask])

    return {
        "r_max": sector_r_max,
        "center": center,
        "num_sectors": num_sectors
    }

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


def build_reference(gray, contour):
    angle = contour_orientation_pca(contour)
    _, contour_rot = rotate_image_and_contour(gray, contour, angle)
    contour_rot = resolve_180_flip(contour_rot)

    hull = cv2.convexHull(contour_rot)
    area = cv2.contourArea(contour_rot)
    hull_area = cv2.contourArea(hull)

    return {
        "contour": contour_rot,
        "hull": hull,
        "area": area,
        "hull_area": hull_area
    }

def build_reference_with_sectors(gray, contour, num_sectors=16):
    base = build_reference(gray, contour)
    if base is None:
        return None

    sectors = compute_sector_profile(
        base["contour"], num_sectors
    )

    base["sectors"] = sectors
    return base

def max_consecutive_true(mask):
    """
    mask: boolean array (circular)
    """
    doubled = np.concatenate([mask, mask])
    max_run = run = 0

    for v in doubled:
        if v:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0

    return min(max_run, len(mask))

def draw_sector_overlay(img, center, ref_r, curr_r, failing):
    cx, cy = map(int, center)
    n = len(ref_r)
    step = 360 / n

    for i in range(n):
        angle_deg = i * step
        angle_rad = np.deg2rad(angle_deg)

        # Reference radius (green)
        x_ref = int(cx + ref_r[i] * np.cos(angle_rad))
        y_ref = int(cy + ref_r[i] * np.sin(angle_rad))

        # Current radius (red if failing)
        x_cur = int(cx + curr_r[i] * np.cos(angle_rad))
        y_cur = int(cy + curr_r[i] * np.sin(angle_rad))

        color = (0, 0, 255) if failing[i] else (0, 255, 0)

        cv2.line(img, (cx, cy), (x_ref, y_ref), (0, 255, 0), 1)
        cv2.line(img, (cx, cy), (x_cur, y_cur), color, 2)




@timed
def short_fill(contour, gray, reference,
                       sector_ratio_thresh=0.92,
                       min_consecutive_sectors=2,
                       area_ratio_thresh=0.95,
                       debug=False):


    angle = contour_orientation_pca(contour)
    img_rot, contour_rot = rotate_image_and_contour(gray, contour, angle)
    contour_rot = resolve_180_flip(contour_rot)

    # --- Global area check ---
    area = cv2.contourArea(contour_rot)
    area_ratio = area / reference["area"]

    # --- Sector profile ---
    sectors = compute_sector_profile(
        contour_rot,
        reference["sectors"]["num_sectors"]
    )

    curr_r = sectors["r_max"]
    ref_r  = reference["sectors"]["r_max"]

    sector_ratios = curr_r / (ref_r + 1e-6)
    failing = sector_ratios < sector_ratio_thresh
    max_run = max_consecutive_true(failing)

    # --- Decision logic ---
    short_fill_fail = (
        area_ratio < area_ratio_thresh or
        max_run >= min_consecutive_sectors
    )

    result = {
        "pass": not short_fill_fail,
        "reason": "short_fill" if short_fill_fail else "ok",
        "scores": {
            "area_ratio": area_ratio,
            "worst_sector_ratio": float(np.min(sector_ratios)),
            "max_consecutive_sectors": int(max_run)
        }
    }

    # --- Debug visualization ---
    if debug:
        overlay = cv2.cvtColor(img_rot, cv2.COLOR_GRAY2BGR)
        draw_sector_overlay(
            overlay,
            sectors["center"],
            ref_r,
            curr_r,
            failing
        )
        result["overlay"] = overlay

    return result

@timed
def pinholes(gray, mask, dbg):
    inv = cv2.bitwise_and(255-gray,255-gray,mask=mask)
    _, thr = cv2.threshold(inv,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cnts,_ = cv2.findContours(thr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    ph=[c for c in cnts if CFG["min_blob_area"]<cv2.contourArea(c)<CFG["max_blob_area"]]
    for c in ph: cv2.drawContours(dbg,[c],-1,(0,0,255),2)
    return {"pass":len(ph)==0,"score":len(ph),"overlay":dbg}

@timed
def blobs(gray, mask, dbg):
    edges=cv2.Canny(gray,80,160)
    edges=cv2.bitwise_and(edges,edges,mask=mask)
    cnts,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    bl=[c for c in cnts if cv2.contourArea(c)>CFG["max_blob_area"]]
    for c in bl: cv2.drawContours(dbg,[c],-1,(255,0,255),2)
    return {"pass":len(bl)==0,"score":len(bl),"overlay":dbg}
