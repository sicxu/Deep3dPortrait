import numpy as np 
from scipy.io import loadmat,savemat
import cv2
from PIL import Image

#######################################################################################
# Auxiliary functions for data preprocessing
# Partially adapted/modified from https://github.com/microsoft/Deep3DFaceReconstruction 
#######################################################################################


#calculating least sqaures problem
def POS(xp, x):
    npts = xp.shape[1]
    # print(npts)

    A = np.zeros([2*npts, 8])

    A[0:2*npts-1:2, 0:3] = x.transpose()
    A[0:2*npts-1:2, 3] = 1

    A[1:2*npts:2, 4:7] = x.transpose()
    A[1:2*npts:2, 7] = 1

    b = np.reshape(xp.transpose(), [2*npts, 1])

    k, _, _, _ = np.linalg.lstsq(A, b)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2

    t = np.stack([sTx, sTy], axis=0)
    return t, s

def img_padding(img, box):
    bbox = box.copy()
    if len(img.shape) != 3:
        res = np.zeros([3*img.shape[0], 3*img.shape[1]])
    else:
        res = np.zeros([3*img.shape[0], 3*img.shape[1], 3])
    res[img.shape[0]: img.shape[0] + img.shape[0],
        img.shape[1]: img.shape[1] + img.shape[1]] = img
    bbox[0] = bbox[0] + img.shape[1]
    bbox[1] = bbox[1] + img.shape[0]
    assert(bbox[0] >= 0 and bbox[1] >= 0)
    return res, bbox

def crop(img, bbox, size=224):
    padded_img, padded_bbox = img_padding(img, bbox)
    crop_img = padded_img[padded_bbox[1]: padded_bbox[1] +
                            padded_bbox[3], padded_bbox[0]: padded_bbox[0] + padded_bbox[2]]
    scale = size / padded_bbox[3]
    return crop_img, scale

def align_img(img, lm, t, s, crp_w=256, crp_h=256, interpolation=cv2.INTER_CUBIC):
    imgw, imgh = img.shape[1], img.shape[0]
    w = np.round(imgw / s).astype(np.int32)
    h = np.round(imgh / s).astype(np.int32)
    t = np.round(t / s).astype(np.int32)
    img = cv2.resize(img.astype(np.uint8), (w, h), 
            interpolation=interpolation).astype(np.float32)
    left = t[0] - crp_w//2
    up = (h - t[1]) - crp_h//2
    bbox = [left, up, crp_w, crp_h]
    cropped_img, _ = crop(img, bbox)
    assert(cropped_img.shape[0] == crp_w)
    # back to raw img s * crop + s * t1
    t1 = np.array([left, h - crp_h - up])
    inv = (s, s * t1)
    lm = (lm - inv[1])/inv[0]
    return cropped_img, inv, lm, bbox

def align_img_(img, lm, t, s):
    imgw, imgh = img.shape[1], img.shape[0]
    M_s = np.array([[1, 0, -t[0] + imgw//2 + 0.5], 
    [0, 1, -imgh//2 + t[1]]], dtype=np.float32)
    img = cv2.warpAffine(img, M_s, (imgw, imgh))
    w, h = int(imgw / s * 100), int(imgh / s * 100)
    img = cv2.resize(img, (w, h))
    lm = np.stack([lm[:, 0] - t[0] + imgw // 2, lm[:, 1] - t[1] + imgh // 2], axis=1) / s * 100

    left, up = w//2 - 112, h//2 - 112
    bbox = [left, up, 224, 224]
    cropped_img, scale_ = crop(img, bbox)
    assert(scale_!=0)
    t1 = np.array([bbox[0], bbox[1]])

    # back to raw img s * crop + s * t1 + t2
    t1 = np.array([w//2 - 112, h//2 - 112])
    scale = s / 100
    t2 = np.array([t[0] - imgw/2, t[1] - imgh / 2])
    inv = (scale/scale_, scale * t1 + t2.reshape([2]))
    return cropped_img, inv

def process_img(img,lm,t,s, scale=102.0):
    w0,h0 = img.size
    img = img.transform(img.size, Image.AFFINE, (1, 0, t[0] - w0/2, 0, 1, h0/2 - t[1]))
    w = (w0/s*scale).astype(np.int32)
    h = (h0/s*scale).astype(np.int32)
    img = img.resize((w,h),resample = Image.BILINEAR)

    # crop the image to 224*224 from image center
    left = (w/2 - 112).astype(np.int32)
    right = left + 224
    up = (h/2 - 112).astype(np.int32)
    below = up + 224

    img = img.crop((left,up,right,below))
    img = np.array(img)
    img = img[:,:,::-1]
    # img = np.expand_dims(img,0)

    lm = np.stack([lm[:,0] - t[0] + w0/2,lm[:,1] - t[1] + h0/2], axis = 1)/s*scale
    lm = lm - np.reshape(np.array([(w/2 - 112),(h/2-112)]),[1,2])

    return img,lm

def facerecon_preprocess(img, lm, lm3D, crp_w=224, crp_h=224):
    lm_idx = np.array([1, 2, 16, 17, 31, 34, 37, 46, 49, 55]) - 1
    lm_ = lm[lm_idx, :]
    t, s = POS(lm_.transpose(), lm3D.transpose())
    cropped_img, inv = align_img_(img, lm.copy(), t, s)
    return cropped_img, inv

def get_invparam(transparam):
    [_, _, s, tx, ty] = transparam
    # res = s * raw + (112 - s * tx, 112 - s * ty)
    # raw = 1/s * res + (-112 / s + tx, -112 / s + ty)
    invparam = (1/s, np.array([-112 / s + tx, -112 / s + ty]))
    return invparam 

def facerecon_preprocess_yu_10p(img, lm, lm3D, crp_w=224, crp_h=224, scale=100.0):
    img = Image.fromarray(img[..., ::-1])
    w0, h0 = img.size
    lm_idx = np.array([1, 2, 16, 17, 31, 34, 37, 46, 49, 55]) - 1
    lm_ = lm[lm_idx, :]
    t, s = POS(lm_.transpose(), lm3D.transpose())
    cropped_img, _ = process_img(img, lm.copy(), t, s, scale)
    trans_params = np.array([w0, h0, scale/s,t[0],t[1]])
    inv = get_invparam(trans_params)
    return cropped_img, inv

def facerecon_preprocess_yu_5p(img,lm,lm3D, scale=102.0):
    img = Image.fromarray(img[..., ::-1])
    w0,h0 = img.size
    # change from image plane coordinates to 3D sapce coordinates(X-Y plane)
    lm = np.stack([lm[:,0],h0 - 1 - lm[:,1]], axis = 1)

    # calculate translation and scale factors using 5 facial landmarks and standard landmarks
    t,s = POS(lm.transpose(),lm3D.transpose())

    # processing the image
    img_new, _ = process_img(img,lm,t,s, scale)
    trans_params = np.array([w0,h0, scale/s,t[0],t[1]])
    inv = get_invparam(trans_params)
    return img_new, inv

def headrecon_preprocess(img, lm, head_center, scale, crp_w=256, crp_h=256):
    cropped_img, inv, crop_lm, _ = align_img(
        img, lm.copy(), head_center.reshape([2]), scale, crp_w, crp_h)
    return cropped_img, inv, crop_lm

def headrecon_preprocess_withmask(img, mask, lm, head_center, scale, crp_w=256, crp_h=256):
    cropped_img, inv, crop_lm, bbox = align_img(
        img, lm.copy(), head_center.reshape([2]), scale, crp_w, crp_h)
    cropped_mask, _, _, _ = align_img(
        mask, lm.copy(), head_center.reshape([2]), scale, crp_w, crp_h, interpolation=cv2.INTER_NEAREST)
    crop_param = np.zeros([5])
    crop_param[0] = scale
    crop_param[1:] = bbox
    return cropped_img, cropped_mask, inv, crop_lm, crop_param