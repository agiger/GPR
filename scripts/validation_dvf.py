import argparse
import os

import csv
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


parser = argparse.ArgumentParser()
parser.add_argument('--root', help='root directory', type=str, required=True)
parser.add_argument('--subdir', help='subdirectory for prediction', type=str, default='validation')
parser.add_argument('--tresh', help='treshold for compactness', type=float, default=0.5)
parser.add_argument('--save', help='flag for saving diff images', action='store_true')
args = parser.parse_args()

if __name__ == "__main__":
    # Read dfs
    dir_predicted = os.path.join(args.root, '{:s}_pred'.format(args.subdir))
    dfs_predicted = sorted([os.path.join(dir_predicted, f) for f in os.listdir(dir_predicted)])
    # print(dfs_predicted)

    dir_gt = os.path.join(args.root, args.subdir)
    dfs_gt = sorted([os.path.join(dir_gt, f) for f in os.listdir(dir_gt)])
    # print(dfs_gt)

    if not len(dfs_predicted) == len(dfs_gt):
        raise Exception('Numbers of predicted DVFs and ground truth DVFs do not match')

    dir_diff = os.path.join(args.root, '{:s}_diff'.format(args.subdir))
    os.makedirs(dir_diff, exist_ok=True)

    tmp = sitk.ReadImage(dfs_gt[0])
    size = tmp.GetSize()
    n_vox = size[0] * size[1] * size[2]
    n_img = len(dfs_gt)
    err = np.empty(shape=(n_vox, n_img))

    for i in range(0, n_img):
        gt = sitk.ReadImage(dfs_gt[i])
        gt_arr = sitk.GetArrayFromImage(gt)

        pred = sitk.ReadImage(dfs_predicted[i])
        pred_arr = sitk.GetArrayFromImage(pred)

        diff_arr = gt_arr - pred_arr
        diff_norm = np.linalg.norm(diff_arr, axis=3)
        err[:, i] = np.reshape(diff_norm, n_vox)

        # 1. Save diff_arr as image
        if args.save:
            diff_img = sitk.GetImageFromArray(diff_arr)
            diff_img.SetSpacing(gt.GetSpacing())
            diff_img.SetOrigin(gt.GetOrigin())
            diff_img.SetDirection(gt.GetDirection())
            sitk.WriteImage(diff_img, os.path.join(dir_diff, 'diff_{:03d}.mha'.format(i)))

    # 2. Compute statistics
    # eliminate zero rows
    err_red = err[~(err == 0).all(1)]
    x = range(0, err_red.shape[1])
    max_err = np.amax(err_red, axis=0)
    min_err = np.amin(err_red, axis=0)
    mean_err = np.mean(err_red, axis=0)
    median_err = np.median(err_red, axis=0)

    per = [50, 75, 90, 95, 99]
    percentiles = np.percentile(err_red.flatten(), per)
    for itr, (tresh, value) in enumerate(zip(per, percentiles)):
        print('{:d}% percentile:\t{:0.4f}'.format(tresh, value))

    errbar= [1, 99, 5, 95, 25, 75]
    errbars= np.percentile(err_red, errbar, axis=0)

    fig1 = plt.figure(figsize=(7.5, 5))
    plt.plot(max_err, label='max error', color='r')
    plt.plot(min_err, label='min error', color='g')
    # plt.plot(mean_err, label='mean error')
    plt.fill_between(x, errbars[0, :], errbars[1, :], edgecolor=(0.91, 0.95, 1), facecolor=(0.91, 0.95, 1),
                     label='01/99 percentiles')
    plt.plot(median_err, label='median error', color='b')
    plt.fill_between(x, errbars[2, :], errbars[3, :], edgecolor=(0.8, 0.9, 1), facecolor=(0.8, 0.9, 1),
                     label='05/95 percentiles')
    plt.fill_between(x, errbars[4, :], errbars[5, :], edgecolor=(0.6, 0.8, 1), facecolor=(0.6, 0.8, 1),
                     label='25/75 percentiles')

    plt.grid()
    # plt.title('Error statistics')
    plt.legend()
    plt.xlabel('sample')
    plt.ylabel('error (mm)')

    # fig2, axs2 = plt.subplots(nrows=5, ncols=10)
    # axs2 = axs2.ravel()
    # for i in range(n_img):
    #     axs2[i].hist(err_red[:, i], 50)
    #     axs2[i].grid()

    color_idx = np.linspace(0, 1, len(per))
    fig2 = plt.figure(figsize=(7.5, 5))
    plt.hist(err_red.flatten(), 50)
    for c, p, pp in zip(color_idx, percentiles, per):
        plt.axvline(p, color=plt.cm.cool(c), lw=2, label='{}th percentile'.format(str(pp)))
    plt.grid()
    plt.legend()
    # plt.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    plt.xlabel('error (mm)')
    plt.ylabel('count')

    # fig1.savefig(os.path.join('/home/alina/Desktop', 'P114_MK_time.pdf'), bbox_inches='tight')
    # fig2.savefig(os.path.join('/home/alina/Desktop', 'P114_MK_hist.pdf'), bbox_inches='tight')

    # PCA: Compactness or "Explained Variance Ratio"
    gpr_dir = os.path.join(args.root, 'gpr')
    # with open(os.path.join(gpr_dir, 'gpr-inputCompactness.csv'), 'r') as input_file:
    #     input_cumsum = list(csv.reader(input_file))
    # with open(os.path.join(gpr_dir, 'gpr-outputCompactness.csv'), 'r') as output_file:
    #     output_cumsum = list(csv.reader(output_file))
    input_cumsum = np.genfromtxt(os.path.join(gpr_dir, 'gpr-inputCompactness.csv'))
    output_cumsum = np.genfromtxt(os.path.join(gpr_dir, 'gpr-outputCompactness.csv'))
    derivative_in = input_cumsum[1:] - input_cumsum[:-1]
    derivative_out = output_cumsum[1:] - output_cumsum[:-1]

    input_sigma = np.genfromtxt(os.path.join(gpr_dir, 'gpr-inputSigma.csv'))
    output_sigma = np.genfromtxt(os.path.join(gpr_dir, 'gpr-outputSigma.csv'))

    n_input = next(i for i, v in enumerate(input_cumsum) if v > args.tresh)
    n_output = next(i for i, v in enumerate(output_cumsum) if v > args.tresh)
    print(n_input, n_output)

    tot_in = sum(input_sigma)
    var_in_exp = [(i / tot_in) for i in sorted(input_sigma, reverse=True)]
    cum_var_in_exp = np.cumsum(var_in_exp)

    tot_out = sum(output_sigma)
    var_out_exp = [(i / tot_out) for i in sorted(output_sigma, reverse=True)]
    cum_var_out_exp = np.cumsum(var_out_exp)

    fig3 = plt.figure()
    x = range(len(input_sigma))
    plt.subplot(2, 1, 1)
    plt.plot(x, input_cumsum, label='from file')
    plt.plot(x, cum_var_in_exp, label='recomputed')
    plt.plot(x[:-1], derivative_in, label='derivative')
    # plt.bar(x, var_in_exp, alpha=0.5, align='center', label='individual explained variance')
    plt.grid()
    plt.title('Input cum sum')
    plt.legend()

    x = range(len(output_sigma))
    plt.subplot(2, 1, 2)
    plt.plot(x, output_cumsum, label='from file')
    plt.plot(x, cum_var_out_exp, label='recomputed')
    plt.plot(x[:-1], derivative_out, label='derivative')
    # plt.bar(x, var_out_exp, alpha=0.5, align='center', label='individual explained variance')
    plt.grid()
    plt.title('Output cum sum')
    plt.legend()

    plt.show()
