import argparse
import os

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-root', help='root directory', type=str, required=True)
parser.add_argument('-save', help='flag for saving diff images', action='store_true')
args = parser.parse_args()

if __name__ == "__main__":
    # Read dfs
    dir_predicted = os.path.join(args.root, 'output', 'validation')
    dfs_predicted = sorted([os.path.join(dir_predicted, f) for f in os.listdir(dir_predicted)])
    print(dfs_predicted)

    dir_gt = os.path.join(args.root, 'output', 'validation_gt')
    dfs_gt = sorted([os.path.join(dir_gt, f) for f in os.listdir(dir_gt)])
    print(dfs_gt)

    if not len(dfs_predicted) == len(dfs_gt):
        raise Exception('Numbers of predicted DVFs and ground truth DVFs do not match')

    dir_diff = os.path.join(args.root, 'output', 'diff')
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
    max_err = np.amax(err_red, axis=0)
    min_err = np.amin(err_red, axis=0)
    mean_err = np.mean(err_red, axis=0)
    median_err = np.median(err_red, axis=0)

    per = [50, 75, 90, 95]
    percentiles = np.percentile(err_red.flatten(), per)

    plt.figure()
    plt.plot(max_err, label='max error')
    plt.plot(min_err, label='min error')
    plt.plot(mean_err, label='mean error')
    plt.plot(median_err, label='median error')
    plt.grid()
    plt.title('Error statistics')
    plt.legend()

    # fig2, axs2 = plt.subplots(nrows=5, ncols=10)
    # axs2 = axs2.ravel()
    # for i in range(n_img):
    #     axs2[i].hist(err_red[:, i], 50)
    #     axs2[i].grid()

    color_idx = np.linspace(0, 1, len(per))
    plt.figure()
    plt.hist(err_red.flatten(), 50)
    for c, p, pp in zip(color_idx, percentiles, per):
        plt.axvline(p, color=plt.cm.cool(c), lw=2, label='{}th percentile'.format(str(pp)))
    plt.grid()
    plt.legend()

    plt.show()
