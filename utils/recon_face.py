import numpy as np
import cv2
from array import array

#######################################################################################
# Auxiliary functions for 3d face reconstruction
# Partially adapted/modified from https://github.com/microsoft/Deep3DFaceReconstruction 
#######################################################################################


def split_coeff(coeff):
    id_coeff = coeff[:,:80] # identity(shape) coeff of dim 80
    ex_coeff = coeff[:,80:144] # expression coeff of dim 64
    tex_coeff = coeff[:,144:224] # texture(albedo) coeff of dim 80
    angles = coeff[:,224:227] # ruler angles(x,y,z) for rotation of dim 3
    gamma = coeff[:,227:254] # lighting coeff for 3 channel SH function of dim 27
    translation = coeff[:,254:] # translation coeff of dim 3

    return id_coeff,ex_coeff,tex_coeff,angles,gamma,translation

## shape computation ## 
def compute_faceshape(coeff, facemodel, inv_params):
    id_coeff,ex_coeff, _, angles, _, translation = split_coeff(coeff)
    face_shape = shape_formation(id_coeff, ex_coeff, facemodel)
    rotation = compute_rotation_matrix(angles)
    # compute vertex projection on image plane (with image sized 224*224)
    face_shape_pose = np.einsum('aij,ajk->aik', face_shape, rotation) + np.reshape(translation,[1,1,3])
    face_projection = projection_layer(face_shape_pose)
    face_projection = face_projection.squeeze(0) * inv_params[0] + inv_params[1].reshape([1, 2])
    landmarks_2d = face_projection[facemodel.keypoints,:]
    return face_shape_pose.squeeze(0), face_projection, landmarks_2d

def compute_center2d(center3d, coeff, facemodel, focal=1015, penter=[112, 112], displace_flag=True, apply_pose=True):
    center3d = center3d.copy()
    _, _, _, angles, _, translation = split_coeff(coeff)
    rotation = compute_rotation_matrix(angles)
    
    if displace_flag:
        # cancel recenter in face recon
        displacement = np.einsum('ij,ajk->aik', 
            np.mean(np.reshape(facemodel.meanshape,[-1,3]), axis = 0, keepdims = True), rotation)
    else:
        displacement = 0
    translation = translation - displacement
    
    if apply_pose:
        center3d = np.einsum('aij,ajk->aik', center3d, rotation)
        center3d = center3d + np.reshape(translation, [np.shape(coeff)[0], 1, 3])
    
    center2d = projection_layer(center3d, focal=focal, penter=penter)
    return center2d, displacement

def projection_layer(face_shape, focal=1015.0, penter=[112.0, 112.0]): # we choose the focal length and camera position empirically
    camera_pos = np.reshape(np.array([0.0,0.0,10.0]),[1,1,3]) # camera position
    reverse_z = np.reshape(np.array([1.0,0,0,0,1,0,0,0,-1.0]),[1,3,3])


    p_matrix = np.concatenate([[focal],[0.0],[penter[0]],[0.0],[focal],[penter[1]],[0.0],[0.0],[1.0]],axis = 0) # projection matrix
    p_matrix = np.reshape(p_matrix,[1,3,3])

    # calculate face position in camera space
    face_shape = np.matmul(face_shape,reverse_z) + camera_pos

    # calculate projection of face vertex using perspective projection
    aug_projection = np.matmul(face_shape, np.transpose(p_matrix,[0,2,1]))
    face_projection = aug_projection[:,:,0:2]/np.reshape(aug_projection[:,:,2],[1,np.shape(aug_projection)[1],1])

    return face_projection

def shape_formation(id_coeff,ex_coeff,facemodel):
    face_shape = np.einsum('ij,aj->ai',facemodel.idBase,id_coeff) + \
                np.einsum('ij,aj->ai',facemodel.exBase,ex_coeff) + \
                facemodel.meanshape

    face_shape = np.reshape(face_shape,[1,-1,3])
    # re-center face shape, yu's setting
    face_shape = face_shape - np.mean(np.reshape(facemodel.meanshape,[1,-1,3]), axis = 1, keepdims = True)
    return face_shape

def compute_rotation_matrix(angles):
    angle_x = angles[:,0][0]
    angle_y = angles[:,1][0]
    angle_z = angles[:,2][0]

    # compute rotation matrix for X,Y,Z axis respectively
    rotation_X = np.array([1.0,0,0,\
        0,np.cos(angle_x),-np.sin(angle_x),\
        0,np.sin(angle_x),np.cos(angle_x)])
    rotation_Y = np.array([np.cos(angle_y),0,np.sin(angle_y),\
        0,1,0,\
        -np.sin(angle_y),0,np.cos(angle_y)])
    rotation_Z = np.array([np.cos(angle_z),-np.sin(angle_z),0,\
        np.sin(angle_z),np.cos(angle_z),0,\
        0,0,1])

    rotation_X = np.reshape(rotation_X,[1,3,3])
    rotation_Y = np.reshape(rotation_Y,[1,3,3])
    rotation_Z = np.reshape(rotation_Z,[1,3,3])

    rotation = np.matmul(np.matmul(rotation_Z,rotation_Y),rotation_X)
    rotation = np.transpose(rotation, axes = [0,2,1])  #transpose row and column (dimension 1 and 2)
    return rotation