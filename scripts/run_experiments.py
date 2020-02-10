import os
import argparse

from main import main
from vtk_mha_converter import converter
from create_filestructure import configure_filestructure


parser = argparse.ArgumentParser()
parser.add_argument('--root', help='root folder containing datasets', type=str, required=True)
parser.add_argument('--config', help='folder containing config files to be analysed', type=str, required=True)
parser.add_argument('--config_filestructure', help='config file for creating filestructure', type=str, default='')
parser.add_argument('--convert_vtk2mha', help='transform vtk image to mha', action='store_true')
args = parser.parse_args()


if __name__ == "__main__":
    if args.config_filestructure:
        configure_filestructure(args.config_filestructure)

    config_files = sorted([os.path.join(args.config, f) for f in os.listdir(args.config) if f.endswith('yaml')])

    datasets = []
    for f in config_files:
        main(f)

        # Extract name of dataset
        data = os.path.splitext(os.path.basename(f))[0][len('config_'):]  # remove 'config_' from filename
        datasets.append(data)

    if args.convert_vtk2mha:
        subdir = os.path.basename(args.config) + '_pred'
        for d in datasets:
            src = os.path.join(args.root, d, 'pairs', 'CT', subdir)
            dest = src + '_mha'
        converter(src, dest)

