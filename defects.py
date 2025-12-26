
import cv2, numpy as np
from timing import timed
from config import CFG

@timed
def circularity(contour, dbg):
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    circ = 4*np.pi*area/(peri*peri+1e-6)
    cv2.drawContours(dbg,[contour],-1,
        (0,255,0) if circ>=CFG["min_circularity"] else (0,0,255),2)
    return {"pass": circ>=CFG["min_circularity"], "score": circ, "overlay": dbg}

@timed
def flash(contour, dbg):
    hull = cv2.convexHull(contour)
    r = (cv2.contourArea(hull)-cv2.contourArea(contour))/cv2.contourArea(hull)
    cv2.drawContours(dbg,[hull],-1,(255,0,0),2)
    return {"pass": r<=CFG["max_flash_ratio"], "score": r, "overlay": dbg}

@timed
def short_fill(contour, dbg):
    hull = cv2.convexHull(contour)
    fill = cv2.contourArea(contour)/cv2.contourArea(hull)
    cv2.drawContours(dbg,[hull],-1,(0,255,255),2)
    return {"pass": fill>=CFG["min_fill_ratio"], "score": fill, "overlay": dbg}

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

@timed
def chips(contour, dbg):
    hi=cv2.convexHull(contour,returnPoints=False)
    d=cv2.convexityDefects(contour,hi)
    if d is None: return {"pass":True,"score":0,"overlay":dbg}
    md=max(x[0][3] for x in d)/256.0
    r=md/np.sqrt(cv2.contourArea(contour))
    return {"pass":r<=CFG["max_chip_depth_ratio"],"score":r,"overlay":dbg}

@timed
def cracks(gray, mask, dbg):
    e=cv2.Canny(gray,60,120)
    e=cv2.bitwise_and(e,e,mask=mask)
    lines=cv2.HoughLinesP(e,1,np.pi/180,50,minLineLength=CFG["min_crack_length"],maxLineGap=5)
    if lines is not None:
        for l in lines:
            x1,y1,x2,y2=l[0]
            cv2.line(dbg,(x1,y1),(x2,y2),(0,0,255),2)
    return {"pass":lines is None,"score":0 if lines is None else len(lines),"overlay":dbg}
