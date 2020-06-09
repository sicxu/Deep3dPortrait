import tensorflow as tf 
import numpy as np
import cv2
from PIL import Image
import os
from scipy.io import loadmat,savemat
from utils.preprocess import POS, headrecon_preprocess_withmask, facerecon_preprocess_yu_5p, facerecon_preprocess
from utils.loader import load_data, load_lm3d, load_center3d, read_facemodel
from utils.recon_face import  compute_center2d, compute_faceshape
from utils.create_renderer import create_renderer_graph
from PIL import Image


def load_facerecon_graph(graph_filename):
    with tf.gfile.GFile(graph_filename,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        input = tf.placeholder(name='input_imgs', shape=[
            None, 224, 224, 3], dtype=tf.float32)
        tf.import_graph_def(graph_def, name='resnet', input_map={
                            'input_imgs:0': input})
        output = graph.get_tensor_by_name('resnet/coeff:0')
    return graph, input, output

def face_recon(input_path, output_path, vis_path=None, s_factor=1.5, focal=1015, center=112, align_nums=10):
    # load BFM
    facemodel = read_facemodel()
    # read standard landmarks for face recon preprocessing
    lm3D = load_lm3d(align_nums)
    # read head center for depth recon preprocessing
    head_center3d = load_center3d()
    
    # create face recon graph
    face_recon_graph, images, coef = load_facerecon_graph('model/model_mask3_white_light.pb')
    face_recon_sess = tf.Session(graph=face_recon_graph)

    # create renderer graph
    depth_render_graph, input_focal, input_center, input_depth, \
        input_vertex, input_tri, output_depthmap = create_renderer_graph()
    render_sess = tf.Session(graph=depth_render_graph)

    imgs_path = [os.path.join(input_path, i) for i in os.listdir(input_path) 
        if i.endswith('png') or i.endswith('jpg') or i.endswith('jpeg')]
    for i, name in enumerate(imgs_path):
        print(i, name.split(os.path.sep)[-1].split('.')[0])
        mask = loadmat(os.path.join(input_path, name.split(os.path.sep)[-1].split('.')[0] + '.mat'))['mask']
        ## load images and corresponding 5 facial landmarks
        if align_nums == 5:
            img, lm = load_data(name, 
                os.path.join(input_path, name.split(os.path.sep)[-1].split('.')[0] + '_detection.txt'))

            lm = lm[-10:].reshape([5, 2])
            input_img, inv_params = facerecon_preprocess_yu_5p(img, lm, lm3D)
        elif align_nums == 10:
            img, lm = load_data(name, 
                os.path.join(input_path, name.split(os.path.sep)[-1].split('.')[0] + '_landmark.txt'))

            lm = lm.reshape([68, 2])
            input_img, inv_params = facerecon_preprocess(img, lm, lm3D)
        
        # recon face
        coeff = face_recon_sess.run(coef, feed_dict={images: np.expand_dims(input_img, 0)})[..., :-1]
        # preprocess input image for depth recon net
        # reproject the reconstructed face to raw image with adjusted focal and center
        f = focal * inv_params[0]
        p_center = inv_params[0] * center + inv_params[1]
        face_shape, face_projection, landmarks_2d = compute_faceshape(coeff, facemodel, inv_params)
        
        # crop the raw image with head center as the image center
        center2d, displacement = compute_center2d(head_center3d, coeff, facemodel, f, p_center)
        _, s =  POS(face_projection.transpose(), facemodel.meanshape.reshape([-1, 3]).transpose())
        crop_img, crop_mask, inv_params_, crop_lm, crop_param = headrecon_preprocess_withmask(img, mask, landmarks_2d, center2d.reshape([2]), s*s_factor/100)
        
        # save processed data
        data = np.zeros([3 + 257 + 136])
        data[0] = f / inv_params_[0]                                            
        data[1: 3] = (p_center - inv_params_[1].reshape([2]))/inv_params_[0]     
        data[3: 260] = coeff.reshape([257])
        data[257: 260] = data[257: 260] - displacement.reshape([3])
        data[260:] = crop_lm.reshape([136])
        face_projection_cropped, _ = compute_center2d(np.expand_dims(face_shape, 0),
            np.expand_dims(data[3:260], 0), facemodel, data[0], data[1:3], displace_flag=False, apply_pose=False)

        # render face depth
        d = 10 - face_shape[:, 2:]
        d = np.tile(np.expand_dims(d, 0), [1, 1, 3])   
        d_map = render_sess.run(output_depthmap, feed_dict={
            input_focal: data[0].reshape([1]),
            input_center: data[1: 3].reshape([1, 1, 2]),
            input_depth: d,
            input_vertex: np.expand_dims(face_shape, 0),
            input_tri: np.expand_dims(facemodel.tri, 0) - 1 # start from 0
        })

        if vis_path:
            cv2.imwrite(os.path.join(vis_path, name.split(os.path.sep)[-1].split('.')[0]+ '.png'),
            crop_img.astype(np.uint8))
        savemat(os.path.join(output_path, name.split(os.path.sep)[-1].split('.')[0]+ '.mat'), 
            {'img': crop_img.astype(np.uint8),
             'mask': crop_mask.astype(np.uint8),
             'crop_param': crop_param.astype(np.float32),
             'face3d': data.astype(np.float32), 
             # 0: focal; [1, 3) center; [3,260): face coeff; [260~396): landmark  
             'face_shape':face_shape.astype(np.float32),
             'face_projection': face_projection_cropped.squeeze(0).astype(np.float32),
             'face_depthmap': d_map[..., 0].squeeze(0), 
             'face_mask': d_map[..., -1].squeeze(0),
             'face_tri': facemodel.tri}, do_compression=True)

    face_recon_sess.close()
    render_sess.close()


if __name__ == '__main__':
    input_path = 'examples'
    save_path = 'output/step1'
    vis_path = 'output/step1/vis'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.isdir(vis_path):
        os.makedirs(vis_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # recon 3d face and prepare the input to depth recon
    face_recon(input_path, save_path, vis_path)

