import os
import csv
import shutil
import argparse
import numpy as np


def empty_dir(path):
    if os.path.exists(path):
        for file in os.listdir(path):
            src = os.path.join(path, file)
            dest = os.path.join(os.path.dirname(path), file)
            shutil.move(src, dest)
    else:
        os.makedirs(path)


def get_us_filename(pairs_ind, file_ind, mode, fname_format):
    us_ind = int(pairs_ind[file_ind, 1])
    if mode == 1:
        dataset_ind = int(pairs_ind[file_ind, 3])
        fname = fname_format.format(dataset_ind, us_ind)
    elif mode == 2:
        fname = fname_format.format(us_ind)
    return fname


def create_pairs(root, split, split_factor, offset=0, mode=1, ar=False,
                 ct_filename='deformationfield_{:03d}.mha', us_filename='us_{:05d}.png'):

    assert len(split) == 3 or len(split) == 5, '{:s}: split indices not correctly defined'.format(root)
    split = [s*split_factor for s in split]
    offset *= split_factor

    # Directories and subdirectories
    pairs_dir = os.path.join(root, 'pairs')
    mr_dir = os.path.join(pairs_dir, 'MR')
    ct_dir = os.path.join(pairs_dir, 'CT')
    us_dir = os.path.join(pairs_dir, 'US')
    ar_dir = os.path.join(pairs_dir, 'AR')

    # MR: remove directory to free space (not used for this analysis)
    if os.path.exists(mr_dir):
        try:
            shutil.rmtree(mr_dir)
            print('Directory {:s} has been removed successfully'.format(mr_dir))
        except Exception as error:
            print(error)
            print('Directory {:s} can not be removed'.format(mr_dir))

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

    if ar and mode == 1:
        ar_train_dir = os.path.join(ar_dir, 'train')
        ar_test_dir = os.path.join(ar_dir, 'test')
        ar_dirs = [ar_train_dir, ar_test_dir]
        empty_dir(ar_train_dir)
        empty_dir(ar_test_dir)
        # Move files from AR to US folder
        files = [os.path.join(ar_dir, f) for f in os.listdir(ar_dir) if f.endswith('png')]
        for f in files:
            shutil.move(f, us_dir)

    # Create US/CT pairs according to list
    file = os.path.join(pairs_dir, 'pairs.csv')
    with open(file, 'r') as f:
        reader = csv.reader(f)
        pairs_ind = list(reader)

    header = pairs_ind[0]
    pairs_ind = np.array(pairs_ind[1:])
    if ar:
        assert len(split) == 5, 'split indices not correctly defined for AR'
        split_ar = split[:2]
        split = split[2:]
        if mode == 1:
            assert pairs_ind.shape[0] % (sum(split) + sum(split_ar) + offset) == 0, 'split indices do not fit dataset'
            p = int(pairs_ind.shape[0]/(sum(split) + sum(split_ar) + offset))  # Order of AR model if any
        elif mode == 2:
            assert pairs_ind.shape[0] % (sum(split) + offset) == 0, 'split indices do not fit dataset'
            p = int(pairs_ind.shape[0]/(sum(split) + offset))  # Order of AR model if any
    print(split_ar, split)

    start_ind = offset
    print(start_ind)
    if ar and mode == 1:  # For stacking: copy US files to AR folders
        for itr_ar_set in range(len(split_ar)):
            for itr_ar_file in range(split_ar[itr_ar_set]*p):
                file_ind = start_ind + itr_ar_file
                fname = get_us_filename(pairs_ind, file_ind, mode, us_filename)
                us_file = os.path.join(us_dir, fname)
                shutil.move(us_file, ar_dirs[itr_ar_set])
            start_ind += split_ar[itr_ar_set]*p
            print(start_ind)

    assert len(split) == 3, 'split indices not correctly defined for US'
    for itr_set in range(len(split)):
        for itr_file in range(split[itr_set]*p):
            if itr_set < 2:  # No CT data for test set
                ct_ind = int(pairs_ind[start_ind + itr_file, 0])
                ct_file = os.path.join(ct_dir, ct_filename.format(ct_ind))
                if itr_file % p == 0:
                    shutil.move(ct_file, ct_dirs[itr_set])

            file_ind = start_ind + itr_file
            fname = get_us_filename(pairs_ind, file_ind, mode, us_filename)
            us_file = os.path.join(us_dir, fname)
            shutil.move(us_file, us_dirs[itr_set])
        start_ind += split[itr_set]*p
        print(start_ind)

    for itr_file in range(offset*p):
        fname = get_us_filename(pairs_ind, itr_file, mode, us_filename)
        us_file = os.path.join(us_dir, fname)
        shutil.move(us_file, us_offset_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='path to dataset', type=str, required=True)
    parser.add_argument('--split', type=int, nargs='+', required=True,
                        help='number of training, validation and test images, respectively')
    parser.add_argument('--split_factor', type=int, default=11, help='number of image pairs per US video')
    parser.add_argument('--offset', type=int, default=0, help='start index of dataset')
    parser.add_argument('--mode', type=int, default=1, help='1: Stacking, 2: Moco')
    parser.add_argument('--ct_filename', type=str, default='deformationfield_{:03d}.mha')
    parser.add_argument('--us_filename', type=str, default='us_{:05d}.png')
    parser.add_argument('--ar', action='store_true', help='use autoregressive model')
    args = parser.parse_args()

    if args.mode == 1:
        split_factor = 1
    else:
        split_factor = args.split_factor

    create_pairs(args.root, args.split, split_factor, args.offset, args.mode, args.ar,
                 args.ct_filename, args.us_filename)
