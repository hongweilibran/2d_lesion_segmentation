from __future__ import print_function
from __future__ import division

import numpy as np
from keras.preprocessing.image import apply_transform, transform_matrix_offset_center

def augmentation(x_0, x_1, y):
    theta = (np.random.uniform(-15, 15) * np.pi) / 180.
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    shear = np.random.uniform(-.1, .1)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])
    zx, zy = np.random.uniform(.9, 1.1, 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    augmentation_matrix = np.dot(np.dot(rotation_matrix, shear_matrix), zoom_matrix)
    transform_matrix = transform_matrix_offset_center(augmentation_matrix, x_0.shape[0], x_0.shape[1])
    x_0 = apply_transform(x_0[..., np.newaxis], transform_matrix, channel_axis=2)
    x_1 = apply_transform(x_1[..., np.newaxis], transform_matrix, channel_axis=2)
    y = apply_transform(y[..., np.newaxis], transform_matrix, channel_axis=2)
    return x_0[..., 0], x_1[..., 0], y[..., 0]

def main():
    return 0

if __name__ == '__main__':
    main()