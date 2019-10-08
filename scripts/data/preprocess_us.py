import os
import argparse

import SimpleITK as sitk
import scipy.ndimage.filters as filters

parser = argparse.ArgumentParser()
parser.add_argument('--src', help='original directory', type=str, required=True)
parser.add_argument('--sigma', help='filter size', type=float, default=2.0)
args = parser.parse_args()


if __name__ == "__main__":
    files = sorted([os.path.join(args.src, f) for f in os.listdir(args.src)])
    dest = '{:s}_blurred'.format(args.src)
    os.makedirs(dest, exist_ok=True)

    for itr, file in enumerate(files):
        img = sitk.GetArrayFromImage(sitk.ReadImage(file))
        img = filters.gaussian_filter(img, args.sigma)
        img = sitk.GetImageFromArray(img)

        fname = os.path.basename(file)
        sitk.WriteImage(img, os.path.join(dest, fname))

        # print(img.shape)