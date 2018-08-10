from __future__ import division
from __future__ import print_function

import numpy as np
import SimpleITK as sitk

from run_test import get_eval_metrics
from models.DRUNet32f import get_model


def main():
    pretrained_model = 'pretrained_models/weights_dru32.h5'
    test_imgs_np_file = 'data/np_data/test_case_148/test/images.npy'
    test_masks_np_file = 'data/np_data/test_case_148/test/masks.npy'

    img_shape = (240, 240, 2)
    num_classes = 9
    model = get_model(img_shape=img_shape, num_classes=num_classes)
    model.load_weights(pretrained_model)

    test_img = np.load(test_imgs_np_file)
    test_masks = np.load(test_masks_np_file)
    test_masks = test_masks[:, :, :, 0]

    img = sitk.ReadImage('data/np_data/test_case_148/test/images.npy')
    # img = sitk.GetArrayFromImage(t1_img)

    pred_mask = model.predict(img)
    pred_mask = pred_mask.argmax(axis=3)
    pred_mask = pred_mask.astype('float32')

    return 0

if __name__ == '__main__':
    main()