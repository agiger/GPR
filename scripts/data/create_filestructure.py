import os
from shutil import copyfile
import argparse
import pydicom

parser = argparse.ArgumentParser()
parser.add_argument('-src_dir', help='source directory with dicom files', type=str, required=True)
parser.add_argument('-dest_dir', help='destination directory for sorted files', type=str, required=True)
args = parser.parse_args()

if __name__ == "__main__":
    # Read files
    files = sorted([os.path.join(args.src_dir, f) for f in os.listdir(args.src_dir)
                    if os.path.isfile(os.path.join(args.src_dir, f))])

    max_sweep_nr = 0
    max_instance_nr = 0
    for itr, file in enumerate(files):
        ds = pydicom.read_file(file)

        # Create destination folder if nonexistent
        dest_protocol = os.path.join(args.dest_dir, ds.ProtocolName)
        dest_series = os.path.join(dest_protocol, str(ds.SeriesNumber))

        os.makedirs(args.dest_dir, exist_ok=True)
        os.makedirs(dest_protocol, exist_ok=True)
        os.makedirs(dest_series, exist_ok=True)

        # Copy file
        dest = os.path.join(dest_series, ('scan%05d.dcm' % ds.InstanceNumber))
        copyfile(file, dest)

        if ds.ProtocolName.startswith('zc_4dmri'):
            if max_sweep_nr < ds.AcquisitionNumber:
                max_sweep_nr = ds.AcquisitionNumber

            if max_instance_nr < ds.InstanceNumber:
                max_instance_nr = ds.InstanceNumber

    f = open(os.path.join(args.dest_dir, 'params.txt'), 'w')
    f.write('n_images: ' + str(max_instance_nr) + '\n')
    f.write('n_sweeps: ' + str(max_sweep_nr) + '\n')
    f.write('n_slices: ' + str(max_instance_nr / max_sweep_nr) + '\n')
