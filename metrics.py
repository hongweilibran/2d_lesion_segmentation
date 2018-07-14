from __future__ import division

from keras import backend as K

def dice_coef(y_true, y_pred):
    '''
    Dice coefficient for multiple categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    smooth = 1e-7
    num_classes = 10

    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=num_classes)[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)

    return K.mean((2. * intersect / (denom + smooth)))


def dice_coef_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1. - dice_coef(y_true, y_pred)