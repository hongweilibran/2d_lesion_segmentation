from __future__ import print_function
from __future__ import division

import click
import json
import os
import numpy as np
from keras.optimizers import Adam

from models.UNet import get_model
from metrics import dice_coef, dice_coef_loss
from run_test import get_eval_metrics


@click.command()
@click.argument('train_imgs_np_file', type=click.STRING)
@click.argument('train_masks_np_file', type=click.STRING)
@click.argument('output_weights_file', type=click.STRING)
@click.option('--pretrained_model', type=click.STRING, default='', help='path to the pretrained model')
@click.option('--test_imgs_np_file', type=click.STRING, default='', help='path to the numpy file of test image')
@click.option('--test_masks_np_file', type=click.STRING, default='', help='path to the numpy file of the test image')
@click.option('--output_test_eval', type=click.STRING, default='', help='path to save results on test case evaluated per epoch of training')
def main(train_imgs_np_file, train_masks_np_file, output_weights_file, pretrained_model='',
         test_imgs_np_file='', test_masks_np_file='', output_test_eval=''):
    assert (test_imgs_np_file != '' and test_masks_np_file != '') or \
           (test_imgs_np_file == '' and test_masks_np_file == ''), \
            'Both test image file and test mask file must be given'

    eval_per_epoch = (test_imgs_np_file != '' and test_masks_np_file != '')
    if eval_per_epoch:
        test_imgs = np.load(test_imgs_np_file)
        test_masks = np.load(test_masks_np_file)

    train_imgs = np.load(train_imgs_np_file)
    train_masks = np.load(train_masks_np_file)

    img_shape = (train_imgs.shape[1], train_imgs.shape[2], 1)
    total_epochs = 2000
    batch_size = 16

    model = get_model(img_shape=img_shape, num_classes=10)
    if pretrained_model != '':
        assert os.path.isfile(pretrained_model)
        model.load_weights(pretrained_model)
    model.compile(optimizer=Adam(lr=(1e-5)), loss=dice_coef_loss, metrics=[dice_coef])

    current_epoch = 1
    history = {}
    history['dsc'] = []
    history['h95'] = []
    history['vs'] = []
    while current_epoch <= total_epochs:
        print('Epoch', str(current_epoch), '/', str(total_epochs))
        model.fit(train_imgs, train_masks, batch_size=batch_size, epochs=1, verbose=True, shuffle=True)
        if eval_per_epoch and current_epoch % 100 == 0:
            model.save_weights(output_weights_file)
            pred_masks = model.predict(test_imgs)
            pred_masks = pred_masks.argmax(axis=3)
            dsc, h95, vs = get_eval_metrics(test_masks[:, :, :, 0], pred_masks)
            history['dsc'].append(dsc)
            history['h95'].append(h95)
            history['vs'].append(vs)
            if output_test_eval != '':
                with open(output_test_eval, 'w+') as outfile:
                    json.dump(history, outfile)

        current_epoch += 1

    model.save_weights(output_weights_file)

    if output_test_eval != '':
        with open(output_test_eval, 'w+') as outfile:
            json.dump(history, outfile)


if __name__ == "__main__":
    main()