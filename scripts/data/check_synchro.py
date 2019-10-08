import os
import argparse

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, required=True)
parser.add_argument('--subdir', type=str, default='train')
parser.add_argument('--flip', action='store_true')
args = parser.parse_args()


if __name__ == "__main__":
    us_dir = os.path.join(args.root, 'pairs', 'US', args.subdir)
    ct_dir = os.path.join(args.root, 'pairs', 'CT', args.subdir)

    us_files = sorted([os.path.join(us_dir, f) for f in os.listdir(us_dir)])
    ct_files = sorted([os.path.join(ct_dir, f) for f in os.listdir(ct_dir)])

    assert len(us_files) == len(ct_files), 'len(us_files) != len(ct_files)'

    us_mean = np.empty(len(us_files))
    ct_mean = np.empty(len(ct_files))

    for itr, (us_file, ct_file) in enumerate(zip(us_files, ct_files)):
        print('Process image\t {:d}/{:d}'.format(itr+1, len(us_files)), end='\r', flush=True)
        us_img = sitk.GetArrayFromImage(sitk.ReadImage(us_file))
        ct_img = sitk.GetArrayFromImage(sitk.ReadImage(ct_file))

        us_mean[itr] = np.mean(us_img)
        ct_mean[itr] = np.mean(np.linalg.norm(ct_img, axis=3))

    # Normalize
    ct_mean = np.clip((ct_mean - np.mean(ct_mean)) / (1e-5 + 3 * np.std(ct_mean)), -1, 1)
    us_mean = np.clip((us_mean - np.mean(us_mean)) / (1e-5 + 3 * np.std(us_mean)), -1, 1)

    if args.flip:
        us_mean *= -1

    plt.figure()
    plt.plot(us_mean, label='US mean')
    plt.plot(ct_mean, label='CT mean')
    plt.grid()
    plt.legend()
    plt.show()
