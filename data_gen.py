from __future__ import division

import os
import click
import numpy as np
import nibabel as nib


def get_2d_patches(input_fold, test_img='', drop_class = [], normilize_per_case=True, output_fold=''):
    """
    Generating 2d patches based on 3d input images.
    Args:
        input_fold: string. Path to the folder with two other folders: images and masks.
                            In turn raw 3d images and masks are stored respectively.
        drop_class: list of integers. List of classes that will be reasigned to background (class 0).
        normilize_per_case: boolean. True when each image per case (3d image) is normilized, False if not.
        test_imgs:  string. Name of the image which is selected for testing among all images.
        output_fold: string. Path to the output folder with numpy files of images and masks.
                             Inside output folder train and test folder will be generated storing their images and masks respectively.
    Returns:
        tuple with 4 numpy files. 1.train images, 2.train masks, 3.test images, 4.test masks.
    """
    assert os.path.isdir(input_fold)
    if output_fold != '':
        assert os.path.isdir(output_fold)
        if output_fold[-1] != '/':
            output_fold += '/'

    if input_fold[-1] != '/':
        input_fold += '/'

    img_files = os.listdir(os.path.join(input_fold, 'images'))

    train_imgs = []
    train_masks = []
    test_imgs = []
    test_masks = []

    for ind, img_file in enumerate(img_files):
        is_train_img = False
        if img_file != test_img:
            is_train_img = True

        img = nib.load(os.path.join(input_fold, 'images/' + img_file)).get_data()
        mask = nib.load(os.path.join(input_fold, 'masks/' + img_file.split('.')[0] + '_mask.nii')).get_data()

        assert img.shape == mask.shape

        img = np.nan_to_num(img)
        mask = np.nan_to_num(mask)

        for elem in drop_class:
            mask[mask == elem] = 0

        if normilize_per_case:
            mean = np.mean(img)
            std = np.std(img)
            img -= mean
            img /= std

        for layer in range(img.shape[2]):
            patch_img = img[:, :, layer]
            patch_mask = mask[:, :, layer]
            patch_img = np.reshape(patch_img, (patch_img.shape[0], patch_img.shape[1], 1))
            patch_mask = np.reshape(patch_mask, (patch_mask.shape[0], patch_mask.shape[1], 1))
            if is_train_img:
                train_imgs.append(patch_img)
                train_masks.append(patch_mask)
            else:
                test_imgs.append(patch_img)
                test_masks.append(patch_mask)

    train_imgs = np.array(train_imgs, dtype='float32')
    train_masks = np.array(train_masks, dtype='float32')
    test_imgs = np.array(test_imgs, dtype='float32')
    test_masks = np.array(test_masks, dtype='float32')

    if output_fold != '':
        if os.path.isdir(os.path.join(output_fold, 'train')) == False:
            os.mkdir(os.path.join(output_fold, 'train'))
        if os.path.isdir(os.path.join(output_fold, 'test')) == False:
            os.mkdir(os.path.join(output_fold, 'test'))
        np.save(os.path.join(output_fold + 'train/', 'images'), train_imgs)
        np.save(os.path.join(output_fold + 'train/', 'masks'), train_masks)
        np.save(os.path.join(output_fold + 'test/', 'images'), test_imgs)
        np.save(os.path.join(output_fold + 'test/', 'masks'), test_masks)

    return (train_imgs, train_masks, test_imgs, test_masks)


@click.command()
@click.argument('input_fold', type=click.STRING)
@click.option('--test_img', default='', type=click.STRING, help='Name of test image among all images')
@click.option('--normilize_per_case', default=True, type=click.BOOL )
@click.option('--output_fold', default='', type=click.STRING, help='Path to the output folder where numpy arrays will be stored')
def main(input_fold, test_img, normilize_per_case, output_fold):
    get_2d_patches(input_fold=input_fold, normilize_per_case=normilize_per_case, drop_class=[], test_img=test_img, output_fold=output_fold)


if __name__ == '__main__':
    main()
