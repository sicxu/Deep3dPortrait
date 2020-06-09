import tensorflow as tf

#######################################################################################
# Auxiliary functions for hair&ear depth prediction
#######################################################################################


def split_data(data):
    focal = data[:, 0:1]
    center = data[:, 1:3]
    face3dparams = data[:, 3:260]
    return focal, center, face3dparams

def get_face_texture(images, facevertex):
    '''
    0.5 for render difference
    '''
    texture = bilinear_sampler(images, facevertex[:, :, 0] - 0.5, tf.cast(
        images.shape[1], tf.float32) - 0.5 - facevertex[:, :, 1])
    return texture

def get_pixel_value(img, xy):
    x = xy[:, :, 0]
    y = xy[:, :, 1]
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')
    x = tf.cast(tf.floor(x), 'int32')
    y = tf.cast(tf.floor(y), 'int32')
    x = tf.clip_by_value(x, zero, max_x)
    y = tf.clip_by_value(y, zero, max_y)
    return gather_pixel_value(img, x, y)

def gather_pixel_value(img, x, y):
    '''
    x = col, y = row
    pixel_value = img(y, x)
    '''
    batch_size = x.shape[0]
    num_points = x.shape[1]
    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1))
    b = tf.tile(batch_idx, (1, num_points))
    indices = tf.stack([b, y, x], 2)
    return tf.gather_nd(img, indices)

def bilinear_sampler(img, x, y):
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = gather_pixel_value(img, x0, y0)
    Ib = gather_pixel_value(img, x0, y1)
    Ic = gather_pixel_value(img, x1, y0)
    Id = gather_pixel_value(img, x1, y1)
    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)

    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=2)
    wb = tf.expand_dims(wb, axis=2)
    wc = tf.expand_dims(wc, axis=2)
    wd = tf.expand_dims(wd, axis=2)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
    return out

def uvd2xyz(uvd, focal, center):
    d = uvd[:, :, 2]
    x = tf.expand_dims(d * (uvd[:, :, 0] - center[:, 0:1]) / focal, axis=2)
    y = tf.expand_dims(d * (uvd[:, :, 1] - center[:, 1:]) / focal, axis=2)
    z = tf.expand_dims(10 - d, axis=2)
    xyz = tf.concat([x, y, z], axis=2)
    return xyz