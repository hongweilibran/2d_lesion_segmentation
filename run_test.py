from __future__ import print_function
from __future__ import division

import click
import numpy as np
import SimpleITK as sitk

from evaluation import getDSC, getHausdorff, getVS


def get_eval_metrics(true_mask, pred_mask):
    true_mask_sitk = sitk.GetImageFromArray(true_mask)
    pred_mask_sitk = sitk.GetImageFromArray(pred_mask)
    dsc = getDSC(true_mask_sitk, pred_mask_sitk)
    h95 = getHausdorff(true_mask_sitk, pred_mask_sitk)
    vs = getVS(true_mask_sitk, pred_mask_sitk)

    result = (dsc, h95, vs)

    return result


@click.command()
@click.argument('true_mask_file', type=click.STRING)
@click.argument('pred_mask_file', type=click.STRING)
def main(true_mask_file, pred_mask_file):
    test_masks = np.load(true_mask_file)
    pred_masks = np.load(pred_mask_file)
    pred_masks = pred_masks.argmax(axis=3)
    test_masks = test_masks[:, :, :, 0]
    dsc, h95, vs = get_eval_metrics(test_masks, pred_masks)
    return (dsc, h95, vs)


if __name__ == '__main__':
    main()