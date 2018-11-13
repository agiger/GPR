import argparse
import os

import pydicom
import SimpleITK as sitk

parser = argparse.ArgumentParser()
parser.add_argument('--src', help='source folder with US dicoms', type=str)
parser.add_argument('--dest', help='destination folder for storing output (vtk images)', type=str)
args = parser.parse_args()

if __name__ == "__main__":
    print(args.src)
    print(args.dest)

    files = sorted([os.path.join(args.src, f) for f in os.listdir(args.src)])
    for itr, file in enumerate(files):
        dcm = pydicom.read_file(file)
        fname = 'video_' + str(dcm.InstanceNumber) + '.vtk'

        print('Write ' + fname + '...')
        image = sitk.ReadImage(file)
        sitk.WriteImage(image, os.path.join(args.dest, fname))

