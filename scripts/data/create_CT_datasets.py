import os
import argparse

import math
import numpy as np
import SimpleITK as sitk


def preprocess_files(src, dest, fmt='mha', tresh=100):
    # Define folder structure and read files
    if not os.path.exists(src):
        raise Exception('No such file or directory: ' + src)

    os.makedirs(dest, exist_ok=True)
    files = sorted([os.path.join(src, f) for f in os.listdir(src) if f.endswith(fmt)])

    # Define ROI
    indices = {
        "x_min": math.inf,
        "x_max": -math.inf,
        "y_min": math.inf,
        "y_max": -math.inf,
        "z_min": math.inf,
        "z_max": -math.inf
    }
    n = min(len(files), tresh)
    for itr, file in enumerate(files[:n]):
        print(file)
        img = sitk.ReadImage(file)
        arr = sitk.GetArrayFromImage(img)
        mag = np.sum(arr, axis=3)

        if mag.max() == 0:
            print("master file: " + file)
        else:
            for s in range(0, mag.shape[0]):
                if mag[s, :, :].max() > 0 and s < indices["z_min"]:
                    indices["z_min"] = s
                    print("new z_min: {}".format(indices["z_min"]))

                s_inv = mag.shape[0] - 1 - s
                if mag[s_inv, :, :].max() > 0 and s_inv > indices["z_max"]:
                    indices["z_max"] = s_inv
                    print("new z_max: {}".format(indices["z_max"]))

            for r in range(0, mag.shape[1]):
                if mag[:, r, :].max() > 0 and r < indices["y_min"]:
                    indices["y_min"] = r
                    print("new y_min: {}".format(indices["y_min"]))

                r_inv = mag.shape[1] - 1 - r
                if mag[:, r_inv, :].max() > 0 and r_inv > indices["y_max"]:
                    indices["y_max"] = r_inv
                    print("new y_max: {}".format(indices["y_max"]))

            for c in range(0, mag.shape[2]):
                if mag[:, :, c].max() > 0 and c < indices["x_min"]:
                    indices["x_min"] = c
                    print("new x_min: {}".format(indices["x_min"]))

                c_inv = mag.shape[2] - 1 - c
                if mag[:, :, c_inv].max() > 0 and c_inv > indices["x_max"]:
                    indices["x_max"] = c_inv
                    print("new x_max: {}".format(indices["x_max"]))

    # Crop image to ROI
    for file in files:
        img = sitk.ReadImage(file)
        sub_img = img[
                  indices["x_min"]:indices["x_max"],
                  indices["y_min"]:indices["y_max"],
                  indices["z_min"]:indices["z_max"]
                  ]
        sitk.WriteImage(sub_img, os.path.join(dest, os.path.basename(file)))

    np.save(os.path.join(src, 'indices_VOI'), indices)


def create_datasets(src, dest, fmt='mha', tresh=100):
    assert os.path.exists(src), 'directory does not exist'
    files = sorted([os.path.join(src, f) for f in os.listdir(src) if f.endswith(fmt)])

    if not os.path.exists(dest):
        os.makedirs(dest)

    if len(files) > 0:
        preprocess_files(src, dest, fmt, tresh)

    for itr, file in enumerate(files):
        os.remove(file)

    print('Files copied: {:d}'.format(len(files)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='data folder', required=True)
    parser.add_argument('--fmt', help='data format', type=str, default='mha')
    parser.add_argument('--tresh', help='max number of images to be analyised for VOI', type=int, default=100)
    args = parser.parse_args()

    dest = os.path.join(args.src, 'pairs', 'CT')
    create_datasets(args.dir, dest, args.fmt, args.tresh)
