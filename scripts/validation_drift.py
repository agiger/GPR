import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--root', help='root directory', type=str, required=True)
parser.add_argument('--label', help='desciption of error data', nargs='+', type=int, default=[1,99,5,95,25,75,50])
args = parser.parse_args()


if __name__ == "__main__":
    data = sorted([os.path.join(args.root, f) for f in os.listdir(args.root) if f.startswith('errbars_drift')])
    ref = os.path.join(args.root, 'errbars.npy')
    ref = np.load(ref)

    data_mean = sorted([os.path.join(args.root, f) for f in os.listdir(args.root) if f.startswith('errbars_mean_drift')])
    ref_mean = os.path.join(args.root, 'errbars_mean.npy')
    ref_mean = np.load(ref_mean)

    tmp = np.load(data[0])
    if not len(args.label) == tmp.shape[0]:
        print("labels not correctly set. Arbitrary labels used")
        label = np.arange(tmp.shape[0])
    else:
        label = args.label
    # label = np.array([label[3], label[6]])

    error_values = np.empty([tmp.shape[0], tmp.shape[1], len(data)+1])
    for itr_d, d in enumerate(data):
        error_values[:, :, itr_d] = np.load(d)
    error_values[:, :, -1] = ref

    tmp_mean = np.load(data_mean[0])
    mean_error_values = np.empty([tmp_mean.shape[0], len(data_mean)+1])
    for itr_d, d in enumerate(data_mean):
        mean_error_values[:, itr_d] = np.load(d)
    mean_error_values[:, -1] = ref_mean

    xticks_for_boxplot = []
    for itr_l, l in enumerate(label):
        plt.figure()
        for itr_d in range(error_values.shape[-1]-1):
            n_train = int(data[itr_d][-7:-4])
            start_ind = int(data[itr_d][-11:-8])
            end_ind = int(start_ind + n_train)
            xticks_for_boxplot.append('{:03d}-{:03d}'.format(start_ind,end_ind))
            plt.plot(error_values[itr_l, :, itr_d], label='trainig data: {:03d}-{:03d}'.format(start_ind, end_ind))
        plt.plot(ref[itr_l, :], label='complete training set')
        xticks_for_boxplot.append('complete')

        plt.grid()
        plt.title('{:d} percentile validation error'.format(l))
        plt.legend()
        plt.xlabel('sample')
        plt.ylabel('error')

    print(error_values.shape)
    for itr_l, l in enumerate(label):
        plt.figure()
        plt.boxplot(error_values[itr_l, :, :])
        plt.grid()
        plt.title('{:d} percentile validation error'.format(l))
        plt.xticks(np.arange(tmp.shape[0]+1), xticks_for_boxplot)

    for itr_l, l in enumerate(label):
        fname = 'boxplots_per{:}'.format(l)
        err = np.squeeze(error_values[itr_l, :, :])
        print(err.shape)
        np.savetxt(os.path.join(args.root, '{:s}.csv'.format(fname)), err, delimiter=',', fmt='%10.4f')
        print(os.path.join(args.root, '{:s}.csv'.format(fname)))

    fname = 'boxplots_mean'
    print(mean_error_values.shape)
    np.savetxt(os.path.join(args.root, 'boxplots_mean.csv'), mean_error_values, delimiter=',', fmt='%10.4f')

    plt.show()
