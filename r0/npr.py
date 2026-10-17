"""Deterministic fake target for engineering tests; never a Ghibli teacher."""
import cv2
import numpy as np

N_CLASSES=7

def edges(rgb, semantic, depth):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    photo=cv2.Canny(gray,50,110)
    boundary=np.zeros_like(semantic)
    boundary[:,1:]=np.maximum(boundary[:,1:],255*(semantic[:,1:]!=semantic[:,:-1]))
    boundary[1:]=np.maximum(boundary[1:],255*(semantic[1:]!=semantic[:-1]))
    return cv2.dilate(cv2.max(photo,boundary),np.ones((2,2),np.uint8))

def stylize(rgb,semantic,depth):
    sm=cv2.bilateralFilter(rgb,7,38,6).astype(np.float32)
    light=sm.mean(axis=2,keepdims=True)
    flat=np.round(sm/36)*36
    warm=flat*1.04+np.array([5,3,-2])
    out=np.uint8(np.clip(warm,0,255))
    out[edges(rgb,semantic,depth)>0]=[42,49,56]
    return out

def pre_stylized_asset_render(rgb,semantic,depth):
    """Route B deterministic coarse material palette (not the training label)."""
    sm=cv2.bilateralFilter(rgb,5,22,4).astype(np.float32)
    out=np.round(sm/64)*64
    # Intentional no line compositing or warm grade: B must still learn corrections.
    return np.uint8(np.clip(out,0,255))
