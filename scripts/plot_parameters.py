import argparse

import csv
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-input_par', help='input parameter file (csv)', type=str)
parser.add_argument('-output_par', help='output parameter file (csv)', type=str)
parser.add_argument('-pred_input_par', help='predicted input parameter file (csv)', type=str)
parser.add_argument('-pred_output_par', help='predicted output parameter file (csv)', type=str)
parser.add_argument('-gt_output_par', help='ground truth output parameter file (csv)', type=str)
parser.add_argument('-input_compactness', help='input compactness file (csv)', type=str)
parser.add_argument('-output_compactness', help='output compactness file (csv)', type=str)
args = parser.parse_args()

if __name__ == "__main__":
    plot_all = True
    plot_max = 5

    # input parameters
    input_pars = []
    with open(args.input_par, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            input_pars.append(row)
    print(input_pars)

    # output parameters
    output_pars = []
    with open(args.output_par, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            output_pars.append(row)
    print(output_pars)

    # predicted input parameters
    pred_input_pars = []
    with open(args.pred_input_par, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            pred_input_pars.append(row)
    print(pred_input_pars)

    # output parameters
    pred_output_pars = []
    with open(args.pred_output_par, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            pred_output_pars.append(row)
    print(pred_output_pars)

    # output parameters
    gt_output_pars = []
    with open(args.gt_output_par, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            gt_output_pars.append(row)
    print(gt_output_pars)

    # input parameters
    input_compactness = []
    with open(args.input_compactness, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            input_compactness.append(row)
    print(input_compactness)

    # output parameters
    output_compactness = []
    with open(args.output_compactness, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            output_compactness.append(row)
    print(output_compactness)

    # Plot
    linew = 0.5
    n_train = len(input_pars)
    n_val = len(pred_input_pars)
    offset = 13
    x1 = list(range(offset, len(input_pars)+offset))
    y1 = list(map(list, zip(*input_pars)))
    if plot_all:
        n_basis_input = len(y1)
    else:
        n_basis_input = plot_max
    if n_basis_input > 1:
        fig1, axs1 = plt.subplots(nrows=n_basis_input, ncols=1)
        fig1.suptitle("Input parameters for training")
        axs1 = axs1.ravel()
        for i in range(0, n_basis_input):
            y1[i] = list(map(float, y1[i]))
            axs1[i].plot(x1, y1[i], label='Parameter c{}'.format(i))
            axs1[i].axhline(0, color='black', lw=linew)
            axs1[i].legend()
    else:
        y1 = list(map(float, y1[0]))
        plt.figure()
        plt.plot(x1, y1, label='Parameter c0')
        plt.axhline(0, color='black', lw=linew)
        plt.legend()
        plt.title("Input parameters for training")

    x2 = list(range(offset, len(output_pars)+offset))
    y2 = list(map(list, zip(*output_pars)))
    if plot_all:
        n_basis_output = len(y2)
    else:
        n_basis_output = plot_max
    if n_basis_output > 1:
        fig2, axs2 = plt.subplots(nrows=n_basis_output, ncols=1)
        fig2.suptitle("Output parameters for training")
        axs2 = axs2.ravel()
        for i in range(0, n_basis_output):
            y2[i] = list(map(float, y2[i]))
            axs2[i].plot(x2, y2[i], label='Parameter c{}'.format(i))
            axs2[i].axhline(0, color='black', lw=linew)
            axs2[i].legend()
    else:
        y2 = list(map(float, y2[0]))
        plt.figure()
        plt.plot(x2, y2, label='Parameter c0')
        plt.axhline(0, color='black', lw=linew)
        plt.legend()
        plt.title("Output parameters for training")

    x3 = list(range(0, len(pred_input_pars)))
    y3 = list(map(list, zip(*pred_input_pars)))
    if n_basis_input > 1:
        fig3, axs3 = plt.subplots(nrows=n_basis_input, ncols=1)
        fig3.suptitle("Input parameters for prediction")
        axs3 = axs3.ravel()
        for i in range(0, n_basis_input):
            y3[i] = list(map(float, y3[i]))
            axs3[i].plot(x3, y3[i], label='Parameter c{}'.format(i))
            axs3[i].axhline(0, color='black', lw=linew)
            axs3[i].legend()
    else:
        y3 = list(map(float, y3[0]))
        plt.figure()
        plt.plot(x3, y3, label='Parameter c0')
        plt.axhline(0, color='black', lw=linew)
        plt.legend()
        plt.title("Input parameters for prediction")

    x4 = list(range(0, len(pred_output_pars)))
    y4a = list(map(list, zip(*pred_output_pars)))
    y4b = list(map(list, zip(*gt_output_pars)))
    if n_basis_output > 1:
        fig4, axs4 = plt.subplots(nrows=n_basis_output, ncols=1)
        fig4.suptitle("Predicted and ground truth output parameters")
        axs4 = axs4.ravel()
        for i in range(0, n_basis_output):
            y4a[i] = list(map(float, y4a[i]))
            y4b[i] = list(map(float, y4b[i]))
            axs4[i].plot(x4, y4a[i], label='Predicted parameter c{}'.format(i))
            axs4[i].plot(x4, y4b[i], label='Ground-truth parameter c{}'.format(i))
            axs4[i].axhline(0, color='black', lw=linew)
            axs4[i].legend()
    else:
        y4a = list(map(float, y4a[0]))
        y4b = list(map(float, y4b[0]))
        plt.figure()
        plt.plot(x4, y4a, label='Predicted parameter c0')
        plt.plot(x4, y4b, label='Ground truth parameter c0')
        plt.axhline(0, color='black', lw=linew)
        plt.legend()
        plt.title("Predicted and ground truth output parameters")

    plt.figure()
    if n_basis_input > 1:
        for i in range(0, n_basis_input):
            max1 = max(y1[i], key=abs)
            y1_norm = [y / max1 for y in y1[i]]
            plt.plot(x1, y1_norm, label='Input parameter c{}'.format(i))
    else:
        max1 = max(y1, key=abs)
        y1_norm = [y / max1 for y in y1]
        plt.plot(x1, y1_norm, label='Input parameter c0')

    if n_basis_output > 1:
        for i in range(0, n_basis_output):
            max2 = max(y2[i], key=abs)
            y2_norm = [y / max2 for y in y2[i]]
            plt.plot(x2, y2_norm, label='Output parameter c{}'.format(i))
    else:
        max2 = max(y2, key=abs)
        y2_norm = [y / max2 for y in y2]
        plt.plot(x1, y2_norm, label='Output parameter c0')

    plt.axhline(0, color='black', lw=0.5)
    plt.title("Input/output parameters for training")
    plt.legend()

    # Compactness
    xSv = list(range(0, len(input_compactness)))
    ySvIn = list(map(list, zip(*input_compactness)))
    ySvIn = list(map(float, ySvIn[0]))
    ySvOut = list(map(list, zip(*output_compactness)))
    ySvOut = list(map(float, ySvOut[0]))

    fig5, axs5 = plt.subplots(nrows=2, ncols=1)
    fig5.suptitle("Compactness")
    axs5 = axs5.ravel()
    axs5[0].plot(xSv, ySvIn, label='Input')
    axs5[0].axhline(0, color='black', lw=linew)
    axs5[0].grid()
    axs5[0].legend()
    axs5[1].plot(xSv, ySvOut, label='Output')
    axs5[1].axhline(0, color='black', lw=linew)
    axs5[1].grid()
    axs5[1].legend()

    # arr_y1 = np.array(y1)
    # print("arr_y1")
    # print(arr_y1)
    # arr_y2 = np.array(y2)
    # print("arr_y2")
    # print(arr_y2)
    # y1_norm = np.linalg.norm(arr_y1, axis=0)
    # y2_norm = np.linalg.norm(arr_y2, axis=0)
    # print(y1_norm)
    # print(y2_norm)

    # plt.figure()
    # y1_norm = y1_norm/np.max(y1_norm)
    # y2_norm = y2_norm/np.max(y2_norm)
    # plt.plot(x1, y1_norm, label='Input parameter c1')
    # plt.plot(x1, y2_norm, label='Output parameter c1')
    # plt.title("Input/output parameters for training")
    # plt.legend()

    plt.show()