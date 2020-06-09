import tensorflow as tf
from .render.mesh_renderer import mesh_renderer
import numpy as np

#######################################################################################
# Auxiliary functions for create renderer
# the mesh_renderer is modified from https://github.com/google/tf_mesh_renderer
#######################################################################################


def create_renderer_graph(v_num=35709, t_num=70789, img_size=256):
    with tf.Graph().as_default() as graph:
        focal = tf.placeholder(dtype=tf.float32, shape=[1])
        center = tf.placeholder(dtype=tf.float32, shape=[1, 1, 2])
        depth = tf.placeholder(dtype=tf.float32, shape=[1, v_num, 3])
        vertex = tf.placeholder(dtype=tf.float32, shape=[1, v_num, 3])
        tri = tf.placeholder(dtype=tf.int32, shape=[1, t_num, 3])
        fov_y = 2 * tf.atan2(img_size//2 * tf.ones_like(focal), focal) / np.pi * 180
        delta_center = tf.concat([(center - img_size//2)/(img_size//2), tf.zeros([center.shape[0], 1, 1])], axis=-1)
        camera_position = tf.constant([0, 0, 10.0])
        camera_lookat = tf.constant([0, 0, 0.0])
        camera_up = tf.constant([0, 1.0, 0])
        light_positions = tf.reshape(tf.constant([0, 0, 1e5]), [1, 1, 3])
        light_intensities = tf.zeros([1, 1, 3])
        depthmap = mesh_renderer(vertex, tri, tf.zeros_like(vertex), depth,
        camera_position=camera_position, camera_lookat=camera_lookat, camera_up=camera_up,
        light_positions=light_positions, light_intensities=light_intensities, 
        image_width=img_size,image_height=img_size,
        fov_y=fov_y, far_clip=30.0, ambient_color=tf.ones([1, 3]), delta_center=delta_center)
    return graph, focal, center, depth, vertex, tri, depthmap
