import os
import numpy as np
import cv2
import tensorflow as tf
from utils.recon_depth import split_data, get_pixel_value, get_face_texture, uvd2xyz
from utils.create_renderer import create_renderer_graph
from scipy.io import savemat, loadmat

_FACE_V_NUM = 35709
_FACE_T_NUM = 70789
_HAIREAR_V_NUM = 28000
_HAIREAR_T_NUM = 54000

def load_depthrecon_graph(graph_filename, image_size=256):
    with tf.gfile.GFile(graph_filename,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        inputs = tf.placeholder(name='inputs', shape=[
            None, image_size, image_size, 5], dtype=tf.float32)
        tf.import_graph_def(graph_def, name='resnet', input_map={
                            'inputs:0': inputs})
        output = graph.get_tensor_by_name('resnet/depth_map:0')
        return graph, inputs, output

def create_shaperecon_graph(image_size=256):
    with tf.Graph().as_default() as graph:
        imgs = tf.placeholder(dtype=tf.float32, shape=[1, image_size, image_size, 3])
        hairear_uv = tf.placeholder(dtype=tf.float32, shape=[1, _HAIREAR_V_NUM, 2])
        hairear_dmap = tf.placeholder(dtype=tf.float32, shape=[1, image_size, image_size, 1])
        face3d_data = tf.placeholder(dtype=tf.float32, shape=[1, 396])
        face_shape2d = tf.placeholder(dtype=tf.float32, shape=[1, _FACE_V_NUM, 2])
        
        focal, center, _ = split_data(face3d_data)
        face_texture = get_face_texture(imgs, face_shape2d)
        hairear_d = get_pixel_value(hairear_dmap, hairear_uv)
        hairear_uv_trans = hairear_uv + 0.5
        hairear_uv_trans = tf.concat([hairear_uv_trans[:, :, 0:1], 
            256 - hairear_uv_trans[:, :, 1:]], axis=-1)
        hairear_uvd = tf.concat([hairear_uv_trans, hairear_d], axis=-1)
        hairear_xyz = uvd2xyz(hairear_uvd, focal, center)
        hairear_texture = get_pixel_value(imgs, hairear_uv)
    return graph, imgs, hairear_uv, hairear_dmap, face3d_data, face_shape2d, face_texture, hairear_xyz, hairear_texture

def depth_recon(data_path, save_path):
    # create face recon graph
    depthrecon_graph, inputs, depth_map = load_depthrecon_graph('model/depth_net.pb')
    depth_recon_sess = tf.Session(graph=depthrecon_graph)

    # create shape recon graph
    shaperecon_graph, input_imgs, input_uv, input_dmap, input_facedata, input_face2d, \
        output_face_texture, output_hairear_xyz, output_hairear_texture = create_shaperecon_graph()
    shape_recon_sess = tf.Session(graph=shaperecon_graph)

    # create renderer graph
    depth_render_graph, input_focal, input_center, input_depth, \
        input_vertex, input_tri, output_depthmap = create_renderer_graph(v_num=_FACE_V_NUM + _HAIREAR_V_NUM, t_num=_FACE_T_NUM + _HAIREAR_T_NUM)
    render_sess = tf.Session(graph=depth_render_graph)

    names = [i for i in os.listdir(data_path) if i.endswith('mat')]
    for i, name in enumerate(names):
        print(i, name.split('.')[0])
        # read and prepare data
        data_input = loadmat(os.path.join(data_path, name))
        imgs_input = data_input['img'].astype(np.float32).reshape([1, 256, 256, 3])
        
        face_d_input = data_input['face_depthmap'].astype(np.float32).reshape([1, 256, 256, 1])
        face_xyz_input = np.expand_dims(data_input['face_projection'].astype(np.float32), 0)
        facewoh_m_input = data_input['facemask_withouthair'].astype(np.float32).reshape([1, 256, 256, 1])
        face3d_data_input = data_input['face3d'].astype(np.float32).reshape([1, 396])

        xy = data_input['points_index']
        uv = np.concatenate([xy[:, 1:], xy[:, :1]], axis=1)
        hairear_uv_input = uv.astype(np.float32).reshape([1, _HAIREAR_V_NUM, 2])
        hairear_m_input = data_input['input_mask'].astype(np.float32).reshape([1, 256, 256, 1])
        
        depth_input = np.concatenate([imgs_input/255, (10 - face_d_input) * facewoh_m_input,
         hairear_m_input], -1)

        # recon hairear depth
        depth_output = depth_recon_sess.run(depth_map, feed_dict={
            inputs: depth_input
        })
        
        # recover head shape from hairear depth
        h_xyz, h_texture, f_texture = shape_recon_sess.run([
            output_hairear_xyz, output_hairear_texture, output_face_texture], 
            feed_dict={
                input_imgs: imgs_input,
                input_dmap: depth_output,
                input_face2d: face_xyz_input,
                input_uv: hairear_uv_input,
                input_facedata: face3d_data_input
        })

        # render head depth
        head_xyz = np.concatenate([
            np.expand_dims(data_input['face_shape'], 0), h_xyz], axis=-2)
        head_d = np.tile(10 - head_xyz[..., -1:], [1, 1, 3])
        head_tri = np.concatenate([
            data_input['face_tri'], data_input['points_tri'] + _FACE_V_NUM], axis=0)
        head_dmap = render_sess.run(output_depthmap, feed_dict={
            input_focal: data_input['face3d'][:, 0].reshape([1]),
            input_center: data_input['face3d'][:, 1:3].reshape([1, 1, 2]),
            input_depth: head_d,
            input_vertex: head_xyz,
            input_tri: np.expand_dims(head_tri, 0) - 1,
        })

        result = {
            'hairear_shape': h_xyz.squeeze(0),
            'hairear_texture': h_texture.squeeze(0),
            'hairear_tri': data_input['points_tri'],
            'face_shape': data_input['face_shape'],
            'face_texture': f_texture.squeeze(0),
            'face_tri': data_input['face_tri'],
            'hairear_index': data_input['points_index'],
            'facemask_withouthair': data_input['facemask_withouthair'],
            'depth': head_dmap[..., 0].squeeze(0), 
            'mask': head_dmap[..., -1].squeeze(0)
        }
        savemat(os.path.join(save_path, name), result, do_compression=True)

    depth_recon_sess.close()
    shape_recon_sess.close()
    render_sess.close()


if __name__ == '__main__':
    data_path = 'output/step2' 
    save_path = 'output/step3' 
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # recon depth and recover the head geometry
    depth_recon(data_path, save_path)
    