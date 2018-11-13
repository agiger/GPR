import os
import argparse

import pydicom
import SimpleITK as sitk

import numpy as np
import math

parser = argparse.ArgumentParser()
parser.add_argument('-src', help='input directory', type=str, required=True)
parser.add_argument('-dest', help='output directory', type=str, required=True)
parser.add_argument('-format', help='file format', type=str, default='mha')
args = parser.parse_args()

if __name__ == "__main__":
    # Define folder structure and read files
    if not os.path.exists(args.src):
        raise Exception('No such file or directory: ' + args.src)

    os.makedirs(args.dest, exist_ok=True)
    files = sorted([os.path.join(args.src, f) for f in os.listdir(args.src) if f.endswith(args.format)])

    # Define ROI
    indices = {
        "x_min": math.inf,
        "x_max": -math.inf,
        "y_min": math.inf,
        "y_max": -math.inf,
        "z_min": math.inf,
        "z_max": -math.inf
    }
    for itr, file in enumerate(files):
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
        sitk.WriteImage(sub_img, os.path.join(args.dest, os.path.basename(file)))

