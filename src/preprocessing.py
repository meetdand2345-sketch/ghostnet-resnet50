import cv2
import numpy as np

def lee_filter(gray, size=7):
    gray=gray.astype(np.float32)
    mean=cv2.blur(gray,(size,size))
    mean_sq=cv2.blur(gray*gray,(size,size))
    var=np.maximum(mean_sq-mean*mean,0)
    noise=max(float(var.mean()),1e-6)
    w=var/(var+noise)
    return np.clip(mean+w*(gray-mean),0,255).astype(np.uint8)

def preprocess(image_bgr):
    gray=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2GRAY)
    filtered=lee_filter(gray)
    clahe=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
    enhanced=clahe.apply(filtered)
    return cv2.cvtColor(enhanced,cv2.COLOR_GRAY2BGR)
