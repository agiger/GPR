import argparse
import os
import sys

import csv
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--root', help='root directory', type=str, required=True)
parser.add_argument('--gpr_subdir', help='gpr subdirectory', type=str, default='pairs/CT/gpr')
parser.add_argument('--gpr_inferenceTime', help='file with computation time for gpr prediction', type=str, default='gpr-latestInferenceTime.txt')
parser.add_argument('--gpr_pcaTime', help='file with computation time for PCA', type=str, default='gpr-latestCompTimePCA.txt')
parser.add_argument('--noshow', help='set flag to not show plots', action='store_true')
args = parser.parse_args()

if __name__ == "__main__":
    # Read all folders
    folders = sorted([os.path.join(args.root, f) for f in os.listdir(args.root) if f.startswith('114CT') or f.startswith('ZL0001')])

    overallCompTime = []
    for itr, folder in enumerate(folders):
        print(folder)
        inferenceTime = np.genfromtxt(os.path.join(folder, args.gpr_subdir, args.gpr_inferenceTime), delimiter=",")
        pcaTime = np.genfromtxt(os.path.join(folder, args.gpr_subdir, args.gpr_pcaTime), delimiter=",")

        # Remove nan at end of array
        inferenceTime = inferenceTime[~np.isnan(inferenceTime)]
        pcaTime = pcaTime[~np.isnan(pcaTime)]
        compTime = inferenceTime + pcaTime
        overallCompTime.append(list(compTime))

        # Printout
        print('mean: {:04f}'.format(np.mean(compTime)))
        print('std: {:04f}'.format(np.std(compTime)))
        print('min: {:04f}'.format(np.min(compTime)))
        print('max: {:04f}\n'.format(np.max(compTime)))

    overallCompTimeArr = np.asarray([item for sublist in overallCompTime for item in sublist])
    print('overallCompTime')
    print('mean: {:04f}'.format(np.mean(overallCompTimeArr)))
    print('std: {:04f}'.format(np.std(overallCompTimeArr)))
    print('min: {:04f}'.format(np.min(overallCompTimeArr)))
    print('max: {:04f}\n'.format(np.max(overallCompTimeArr)))


