import argparse
import os

import csv
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib2tikz

parser = argparse.ArgumentParser()
parser.add_argument('--root', help='root directory', type=str, required=True)
parser.add_argument('--subdir', help='subdirectory for prediction', type=str, default='train')
parser.add_argument('--suffix', help='subdirectory suffix for further specification', type=str, default='')
parser.add_argument('--noshow', help='set flag to not show plots', action='store_true')
args = parser.parse_args()

if __name__ == "__main__":
    dir_gt = os.path.join(args.root, args.subdir)
    dfs_gt = sorted([os.path.join(dir_gt, f) for f in os.listdir(dir_gt)])

    n_img = len(dfs_gt)
    mean_dvf = np.empty(shape=(4, n_img))

    spacing = [3, 2.54, 2.54]
    # spacing = [3, 1.95, 1.95]

    for i in range(0, n_img):
        gt = sitk.ReadImage(dfs_gt[i])
        gt_arr = sitk.GetArrayFromImage(gt)
        for dim in range(3):
            gt_arr[:, :, :, dim] *= spacing[dim]

        mean_dvf[0, i] = np.mean(gt_arr[:, :, :, 0])
        mean_dvf[1, i] = np.mean(gt_arr[:, :, :, 1])
        mean_dvf[2, i] = np.mean(gt_arr[:, :, :, 2])
        mean_dvf[3, i] = np.mean(np.linalg.norm(gt_arr, axis=3))

    f = 1.25  # Hz
    x = np.arange(0, n_img)
    x_time = np.divide(x, f)
    fig1 = plt.figure(figsize=(7.5, 5))
    plt.subplot(4, 1, 1)
    plt.plot(x_time, mean_dvf[0, :]-np.mean(mean_dvf[0, :]))
    plt.grid()
    plt.title('Mean deformation field in LR')
    plt.xlabel('time [s]')
    plt.ylabel('deformation [mm]')

    plt.subplot(4, 1, 2)
    plt.plot(x_time, mean_dvf[1, :]-np.mean(mean_dvf[1, :]))
    plt.grid()
    plt.title('Mean deformation field in AP')
    plt.xlabel('time [s]')
    plt.ylabel('deformation [mm]')

    plt.subplot(4, 1, 3)
    plt.plot(x_time, mean_dvf[2, :]-np.mean(mean_dvf[2, :]))
    plt.grid()
    plt.title('Mean deformation field in SI')
    plt.xlabel('time [s]')
    plt.ylabel('deformation [mm]')

    plt.subplot(4, 1, 4)
    plt.plot(x_time, mean_dvf[3, :]-np.mean(mean_dvf[3, :]))
    plt.grid()
    plt.title('Mean deformation field magnitude')
    plt.xlabel('time [s]')
    plt.ylabel('deformation [mm]')

    matplotlib2tikz.save(os.path.join(args.root, 'resp_motion_{:s}.tex'.format(args.subdir)))

    if not args.noshow:
        plt.show()
