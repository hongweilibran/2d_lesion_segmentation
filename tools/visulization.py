from __future__ import print_function
from __future__ import division

import click
import os
import json
import numpy as np
import matplotlib.pyplot as plt


def visualize_test_evaluation(input_file, output_folder):
    assert os.path.isfile(input_file)

    with open(input_file) as infile:
        eval_dict = json.load(infile)

    dsc_list = eval_dict['dsc']
    h95_list = eval_dict['h95']
    vs_list = eval_dict['vs']

    dsc_avg_list = [np.nanmean(np.array(item.values(), dtype='float32')) for item in dsc_list]
    h95_avg_list = [np.nanmean(np.array(item.values(), dtype='float32')) for item in h95_list]
    vs_avg_list = [np.nanmean(np.array(item.values(), dtype='float32')) for item in vs_list]

    x_axis = list(range(len(dsc_avg_list)))

    plt.plot(x_axis, dsc_avg_list, 'r', x_axis, h95_avg_list, 'b', x_axis, vs_avg_list, 'g')
    plt.title('Dice Coeffient and Volume similarity')
    plt.savefig(os.path.join(output_folder, 'dice_vs_graph.png'))
    plt.close()

    plt.plot(x_axis, h95_avg_list, 'b')
    plt.title('Hausdorff distance')
    plt.savefig(os.path.join(output_folder, 'h95.png'))
    plt.close()

    return 0

@click.command()
@click.argument('input_file', type=click.STRING)
@click.argument('output_folder', type=click.STRING)
def main(input_file, output_folder):
    visualize_test_evaluation(input_file, output_folder)
    return 0

if __name__ == '__main__':
    main()