import os
import numpy as np
import cv2
from scipy.io import loadmat, savemat
from scipy.spatial import Delaunay
from utils.loader import load_mask
from utils.face_parsing import split_segmask
from utils.construct_triangles import remove_small_area, filter_tri, padding_tri


def prepare_mask(input_path, save_path, mask_path, vis_path=None, filter_flag=True, padding_flag=True):
    names = [i for i in os.listdir(input_path) if i.endswith('mat')]
    for i, name in enumerate(names):
        print(i, name.split('.')[0])
        # get input mask
        data = loadmat(os.path.join(input_path, name))
        render_mask = data['face_mask']
        seg_mask = load_mask(os.path.join(mask_path, name))
        face_segmask, hairear_mask, _ = split_segmask(seg_mask)
        face_remain_mask = np.zeros_like(face_segmask)
        face_remain_mask[(face_segmask - render_mask) == 1] = 1
        stitchmask = np.clip(hairear_mask + face_remain_mask, 0, 1)
        stitchmask = remove_small_area(stitchmask)
        facemask_withouthair = render_mask.copy()
        facemask_withouthair[(render_mask + hairear_mask) == 2] = 0

        if vis_path:
            cv2.imwrite(os.path.join(vis_path, name.split('.mat')[0] + '.png'),
            (data['img'].astype(np.float32) * np.expand_dims(hairear_mask, 2).astype(np.float32)).astype(np.uint8))

        # get triangle
        points_index = np.where(stitchmask == 1)
        points = np.array([[points_index[0][i], points_index[1][i]]
                            for i in range(points_index[0].shape[0])])
        tri = Delaunay(points).simplices.copy()
        if filter_flag :
            # constrain the triangle size
            tri = filter_tri(tri, points)
        if padding_flag:
            # padding the points and triangles to predefined nums 
            points, tri = padding_tri(points.copy(), tri.copy())
        data['input_mask'] = stitchmask
        data['points_tri'] = tri + 1 # start from 1
        data['points_index'] = points
        data['facemask_withouthair'] = facemask_withouthair
        savemat(os.path.join(save_path, name), data, do_compression=True)


if __name__ == '__main__':
    input_path = 'output/step1'
    mask_path = 'output/step1'
    save_path = 'output/step2'
    vis_path = 'output/step2/vis'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.isdir(vis_path):
        os.makedirs(vis_path)
    # prepare the input mask
    prepare_mask(input_path, save_path, mask_path, vis_path)
