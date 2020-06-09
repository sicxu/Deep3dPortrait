import numpy as np
from skimage.morphology import label, remove_small_objects
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
import cv2

#######################################################################################
# Auxiliary functions for triangulation
#######################################################################################

def remove_small_area(mask, thres=20):
    labels = label(mask.astype(np.int32), connectivity=1)
    mask = remove_small_objects(labels.astype(np.bool), thres, connectivity=1)
    return mask.astype(np.float32)

def dis(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def filter_tri(tri, points, thres=np.sqrt(5)):
    tri_list = []
    for i in range(tri.shape[0]):
        dis_ab = dis(points[tri[i][0]], points[tri[i][1]])
        dis_bc = dis(points[tri[i][1]], points[tri[i][2]])
        dis_ac = dis(points[tri[i][0]], points[tri[i][2]])
        max_dis = np.max([dis_ab, dis_bc, dis_ac])
        if max_dis > thres: continue
        tri_list.append(tri[i])
    return np.array(tri_list)

def padding_tri(points, tri_list, max_num=28000, max_list=54000, OUTLIER=1000):
    assert points.shape[0] <= max_num and tri_list.shape[0] <= max_list
    padding = OUTLIER * np.ones([max_num - points.shape[0], 2])
    padded_points = np.concatenate([points, padding], axis=0).astype(np.int32)
    padding_list = np.tile(np.array(
        [max_num-3, max_num-2, max_num-1]).reshape([1, 3]), [max_list - tri_list.shape[0], 1])
    padding_list = np.concatenate(
        [tri_list,padding_list], axis=0).astype(np.int32)
    return padded_points, padding_list


def find_boundary_ind(hair_shape, face_shape, points_index, border, mask):

    # find the boundary between rendered face and hair on hair in image plane
    boundary = cv2.dilate(mask.astype(np.uint8), 
        np.uint8(np.ones((3, 3))), iterations=1).astype(np.float32) - mask
    boundary = remove_small_area(boundary, thres=200)
    index = np.where(boundary == 1)
    boundary_ind = np.concatenate(
        [np.expand_dims(index[0], axis=1), np.expand_dims(index[1], axis=1)], axis=1)
    
    # filter the boundary points on hair 
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(points_index)
    dist, idx = nbrs.kneighbors(boundary_ind)
    dist, idx = dist.squeeze(1), idx.squeeze(1)
    match_boundary_ind = idx[dist<2]
    _, tmp = np.unique(match_boundary_ind, return_index=True)
    hair_boundary_ind = match_boundary_ind[np.sort(tmp)]

    # find the boundary between face and hair on face in 3d space
    hair_boundary = hair_shape[hair_boundary_ind[:]].copy()
    nbrs3d = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(face_shape)
    dist3d, idx3d = nbrs3d.kneighbors(hair_boundary)
    dist3d, idx3d = dist3d.squeeze(1), idx3d.squeeze(1)
    match_boundary_ind = np.concatenate([idx3d, border])
    _, tmp = np.unique(match_boundary_ind, return_index=True)
    face_boundary_ind = match_boundary_ind[np.sort(tmp)]
    
    return hair_boundary_ind, face_boundary_ind

def construct_triangle(hair_xyz, hair_texture, face_xyz, face_texture, hb_ind, fb_ind):
    xyz = np.concatenate([face_xyz[fb_ind[:]], hair_xyz[hb_ind[:]]], axis=0)
    texture = np.concatenate([face_texture[fb_ind[:]], hair_texture[hb_ind[:]]], axis=0)
    tri = Delaunay(xyz[:, :2])
    tri_list = tri.simplices.copy()
    tri_list = filter_tri(tri_list, xyz, thres=0.05) + 1
    return xyz, texture, tri_list
