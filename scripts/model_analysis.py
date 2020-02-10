import argparse
import os
import sys

import csv
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--root', help='root directory', type=str, required=True)
parser.add_argument('--suffix', help='subdirectory suffix for further specification', type=str, default='')
parser.add_argument('--tresh', help='treshold for compactness', type=float, default=0.5)
parser.add_argument('--noshow', help='set flag to not show plots', action='store_true')
args = parser.parse_args()

if __name__ == "__main__":
    # PCA: Compactness or "Explained Variance Ratio"
    gpr_dir = os.path.join(args.root, 'gpr{:s}'.format(args.suffix))
    input_cumsum = np.genfromtxt(os.path.join(gpr_dir, 'gpr-inputCompactness.csv'))
    output_cumsum = np.genfromtxt(os.path.join(gpr_dir, 'gpr-outputCompactness.csv'))

    derivative_in = input_cumsum[1:] - input_cumsum[:-1]
    derivative_out = output_cumsum[1:] - output_cumsum[:-1]
    derivative_in = np.insert(derivative_in, 0, input_cumsum[0])
    derivative_out = np.insert(derivative_out, 0, output_cumsum[0])

    n_input = next(i for i, v in enumerate(input_cumsum) if v > args.tresh)
    n_output = next(i for i, v in enumerate(output_cumsum) if v > args.tresh)
    print(n_input, n_output)
    print(input_cumsum[1], output_cumsum[1])

    fig0 = plt.figure()
    x = np.arange(input_cumsum.shape[0])
    plt.subplot(2, 1, 1)
    plt.plot(x, input_cumsum, label='Explained variance')
    plt.plot(x, derivative_in, label='Derivative')
    # plt.bar(x, var_in_exp, alpha=0.5, align='center', label='individual explained variance')
    plt.grid()
    plt.title('Input')
    plt.legend()

    x = range(output_cumsum.shape[0])
    plt.subplot(2, 1, 2)
    plt.plot(x, output_cumsum, label='Explained variance')
    plt.plot(x, derivative_out, label='Derivative')
    # plt.bar(x, var_out_exp, alpha=0.5, align='center', label='individual explained variance')
    plt.grid()
    plt.title('Output')
    plt.legend()

    if not args.noshow:
        plt.show()
