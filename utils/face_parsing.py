import numpy as np
from scipy.io import loadmat

#######################################################################################
# Auxiliary functions for face segmentation
# for face parsing, please refer to https://arxiv.org/pdf/1906.01342.pdf
#######################################################################################


def faceparsing():
    # return a label with 5 classes:
    # 0: bg 1: face 2: hair 3: left ear 4: right ear 5(optional): inner mouth
    return NotImplemented

def split_segmask(mask):
    face_mask, hairear_mask, mouth_mask = np.zeros_like(mask), np.zeros_like(mask), np.zeros_like(mask)
    face_mask[mask==1] = 1
    face_mask[mask==5] = 1
    hairear_mask[mask==2] = 1
    hairear_mask[mask==3] = 1
    hairear_mask[mask==4] = 1
    mouth_mask[mask==5] = 1
    return face_mask, hairear_mask, mouth_mask

