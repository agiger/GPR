import os
import csv
import shutil
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--root', help='path to dataset', type=str, required=True)
parser.add_argument('--split', type=int, nargs='+', required=True,
                    help='number of training, validation and test images, respectively')
parser.add_argument('--offset', type=int, default=0, help='start index of dataset')
parser.add_argument('--ct_filename', type=str, default='deformationfield_{:03d}.mha')
parser.add_argument('--us_filename', type=str, default='us_{:05d}.png')
parser.add_argument('--mode', type=int, default=1, help='1: Stacking, 2: Moco')
args = parser.parse_args()


def empty_dir(path):
    if os.path.exists(path):
        for file in os.listdir(path):
            src = os.path.join(path, file)
            dest = os.path.join(os.path.dirname(path), file)
            shutil.move(src, dest)
    else:
        os.makedirs(path)


if __name__ == "__main__":
    # Directories and subdirectories
    pairs_dir = os.path.join(args.root, 'pairs')
    mr_dir = os.path.join(pairs_dir, 'MR')
    ct_dir = os.path.join(pairs_dir, 'CT')
    us_dir = os.path.join(pairs_dir, 'US')

    # MR: remove directory to free space (not used for this analysis)
    if os.path.exists(mr_dir):
        try:
            shutil.rmtree(mr_dir)
            print('Directory {:s} has been removed successfully'.format(mr_dir))
        except Exception as error:
            print(error)
            print('Directory {:s} can not be removed'.format(mr_dir))
    else:
        print('No MR directory')

    # CT: Move existing files to parent folder
    ct_train_dir = os.path.join(ct_dir, 'train')
    ct_val_dir = os.path.join(ct_dir, 'validation')
    ct_test_dir = os.path.join(ct_dir, 'test')
    ct_dirs = [ct_train_dir, ct_val_dir, ct_test_dir]

    empty_dir(ct_train_dir)
    empty_dir(ct_val_dir)
    empty_dir(ct_test_dir)

    # US: Move existing files to parent folder
    us_train_dir = os.path.join(us_dir, 'train')
    us_val_dir = os.path.join(us_dir, 'validation')
    us_test_dir = os.path.join(us_dir, 'test')
    us_offset_dir = os.path.join(us_dir, 'offset')
    us_dirs = [us_train_dir, us_val_dir, us_test_dir]

    empty_dir(us_train_dir)
    empty_dir(us_val_dir)
    empty_dir(us_test_dir)
    empty_dir(us_offset_dir)

    # Create US/CT pairs according to list
    file = os.path.join(pairs_dir, 'pairs.csv')
    with open(file, 'r') as f:
        reader = csv.reader(f)
        pairs_ind = list(reader)

    header = pairs_ind[0]
    pairs_ind = np.array(pairs_ind[1:])

    if args.mode == 1:  # US index == MR index for stacking
        pairs_ind[:, 1] = pairs_ind[:, 0]

    assert sum(args.split) + args.offset == pairs_ind.shape[0], 'split indices do not fit dataset'

    start_ind = args.offset
    print(start_ind)
    for itr_set in range(len(args.split)):
        for itr_file in range(args.split[itr_set]):
            if itr_set < 2:  # No CT data for test set
                ct_ind = int(pairs_ind[start_ind + itr_file, 0])
                ct_file = os.path.join(ct_dir, args.ct_filename.format(ct_ind))
                shutil.move(ct_file, ct_dirs[itr_set])

            us_ind = int(pairs_ind[start_ind + itr_file, 1])
            us_file = os.path.join(us_dir, args.us_filename.format(us_ind))
            shutil.move(us_file, us_dirs[itr_set])
        start_ind += args.split[itr_set]
        print(start_ind)

    for itr_file in range(args.offset):
        us_ind = int(pairs_ind[itr_file, 1])
        us_file = os.path.join(us_dir, args.us_filename.format(us_ind))
        shutil.move(us_file, us_offset_dir)


