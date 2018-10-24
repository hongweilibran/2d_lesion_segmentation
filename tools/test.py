import numpy as np
import SimpleITK as sitk
from run_test import get_eval_metrics

def main():
    mask = '/home/zhygallo/zhygallo/tum/GuidedResearch/miccai/training_corrected/070/segm.nii.gz'
    pred_mask = '/home/zhygallo/zhygallo/tum/GuidedResearch/final_submission/python/output/result.nii.gz'

    mask = sitk.ReadImage(mask)
    pred_mask = sitk.ReadImage(pred_mask)

    mask = sitk.GetArrayFromImage(mask)
    pred_mask = sitk.GetArrayFromImage(pred_mask)

    dsc, h95, vs = get_eval_metrics(mask, pred_mask)

    return 0

if __name__ == "__main__":
    main()