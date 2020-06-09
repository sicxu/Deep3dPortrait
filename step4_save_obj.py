import os
from scipy.io import loadmat
from utils.construct_triangles import remove_small_area, filter_tri, find_boundary_ind, construct_triangle
from utils.loader import load_boundary_ind


def save_obj(data_path, save_path, stitch=True):
    names = [i for i in os.listdir(data_path) if i.endswith('.mat')]
    border_index = load_boundary_ind()

    for i, name in enumerate(names):
        print(i, name.split('.')[0])
        data = loadmat(os.path.join(data_path, name))
        hair_xyz = data['hairear_shape']
        hair_texture = data['hairear_texture']
        hair_tri = data['hairear_tri']
        
        face_xyz = data['face_shape']
        face_texture = data['face_texture']
        face_tri = data['face_tri']

        mask =  data['facemask_withouthair']

        # define triangles between face and other parts of head
        if stitch:
            hb_ind, fb_ind = find_boundary_ind(
                hair_xyz, face_xyz, data['hairear_index'], border_index, mask
            )

            border_xyz, border_texture, border_tri = construct_triangle(
                hair_xyz, hair_texture, face_xyz, face_texture, hb_ind, fb_ind
            )

        with open(os.path.join(save_path, name.split('.')[0] + '.obj'), 'w') as f:
            for i in range(face_xyz.shape[0]):
                f.write('v %f %f %f %f %f %f\n' %(*face_xyz[i, :], *face_texture[i, ::-1]))
            for i in range(hair_xyz.shape[0]):
                f.write('v %f %f %f %f %f %f\n' %(*hair_xyz[i, :], *hair_texture[i, ::-1]))
            if stitch:
                for i in range(border_xyz.shape[0]):
                    f.write('v %f %f %f %f %f %f\n' %(*border_xyz[i, :], *border_texture[i, ::-1]))

            for i in range(face_tri.shape[0]):
                f.write('f {} {} {}\n'.format(*face_tri[i, :]))
            for i in range(hair_tri.shape[0]):
                f.write('f {} {} {}\n'.format(*hair_tri[i, :] + face_xyz.shape[0]))
            if stitch:
                for i in range(border_tri.shape[0]):
                    f.write('f {} {} {}\n'.format(*border_tri[i, :] + face_xyz.shape[0] + hair_xyz.shape[0]))
                

if __name__ == '__main__':
    data_path = 'output/step3'
    save_path = 'output/step4'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    # save the recovered geometry as obj files
    save_obj(data_path, save_path, True)
