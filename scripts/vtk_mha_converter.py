import argparse
import os
import SimpleITK as sitk


def converter(src, dest):
    os.makedirs(dest, exist_ok=True)
    files = sorted([os.path.join(src, f) for f in os.listdir(src) if f.endswith('vtk')])

    for i, file in enumerate(files):
        img = sitk.ReadImage(file)
        fname = os.path.basename(file)
        fname = fname[:-3] + 'mha'
        sitk.WriteImage(img, os.path.join(dest, fname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', help='source directory', type=str, required=True)
    parser.add_argument('-dest', help='destination', type=str, required=True)
    args = parser.parse_args()

    converter(args.src, args.dest)
