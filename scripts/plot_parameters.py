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
    plot_nInputC = 1
    plot_nOutputC = 1

    # input parameters
    input_pars = np.loadtxt(args.input_par, delimiter=',')
    print(input_pars.shape)

    # output parameters
    output_pars = np.loadtxt(args.output_par, delimiter=',')
    print(output_pars.shape)

    # input parameters for prediction
    pred_input_pars = np.loadtxt(args.pred_input_par, delimiter=',')
    print(pred_input_pars.shape)

    # predicted output parameters
    pred_output_pars = np.loadtxt(args.pred_output_par, delimiter=',')
    print(pred_output_pars.shape)

    # ground truth output parameters
    gt_output_pars = np.loadtxt(args.gt_output_par, delimiter=',')
    print(gt_output_pars.shape)

    # input compactness
    input_compactness = np.loadtxt(args.input_compactness, delimiter=',')
    print(input_compactness.shape)

    # output compactness
    output_compactness = np.loadtxt(args.output_compactness, delimiter=',')
    print(output_compactness.shape)

    # Plot
    linew = 0.5
    offset = 13
    nTrainFiles = input_pars.shape[1]
    nTestFiles = pred_input_pars.shape[1]
    xTrain = np.arange(offset, nTrainFiles+offset)
    xTest = np.arange(0, nTestFiles)

    # Fig 1: Input parameters for training
    if plot_nInputC > 1:
        fig1, axs1 = plt.subplots(nrows=plot_nInputC, ncols=1)
        fig1.suptitle("Input parameters for training")
        axs1 = axs1.ravel()
        for i in range(0, plot_nInputC):
            axs1[i].plot(xTrain, input_pars[i, :], label='Parameter c{}'.format(i))
            axs1[i].axhline(0, color='black', lw=linew)
            axs1[i].legend()
    else:
        plt.figure()
        plt.plot(xTrain, input_pars[0, :], label='Parameter c0')
        plt.axhline(0, color='black', lw=linew)
        plt.legend()
        plt.title("Input parameters for training")

    # Fig 2: Output parameters for training
    if plot_nOutputC > 1:
        fig2, axs2 = plt.subplots(nrows=plot_nOutputC, ncols=1)
        fig2.suptitle("Output parameters for training")
        axs2 = axs2.ravel()
        for i in range(0, plot_nOutputC):
            axs2[i].plot(xTrain, output_pars[i, :], label='Parameter c{}'.format(i))
            axs2[i].axhline(0, color='black', lw=linew)
            axs2[i].legend()
    else:
        plt.figure()
        plt.plot(xTrain, output_pars[0, :], label='Parameter c0')
        plt.axhline(0, color='black', lw=linew)
        plt.legend()
        plt.title("Output parameters for training")

    # Fig 3: Input parameters for prediction
    if plot_nInputC > 1:
        fig3, axs3 = plt.subplots(nrows=plot_nInputC, ncols=1)
        fig3.suptitle("Input parameters for prediction")
        axs3 = axs3.ravel()
        for i in range(0, plot_nInputC):
            axs3[i].plot(xTest, pred_input_pars[i, :], label='Parameter c{}'.format(i))
            axs3[i].axhline(0, color='black', lw=linew)
            axs3[i].legend()
    else:
        plt.figure()
        plt.plot(xTest, pred_input_pars[0, :], label='Parameter c0')
        plt.axhline(0, color='black', lw=linew)
        plt.legend()
        plt.title("Input parameters for prediction")

    # Fig 4: Predicted output parameters
    if plot_nOutputC > 1:
        fig4, axs4 = plt.subplots(nrows=plot_nOutputC, ncols=1)
        fig4.suptitle("Predicted and ground truth output parameters")
        axs4 = axs4.ravel()
        for i in range(0, plot_nOutputC):
            axs4[i].plot(xTest, gt_output_pars[i, :], label='Ground-truth parameter c{}'.format(i))
            axs4[i].plot(xTest, pred_output_pars[i, :], label='Predicted parameter c{}'.format(i))
            axs4[i].axhline(0, color='black', lw=linew)
            axs4[i].legend()
    else:
        plt.figure()
        plt.plot(xTest, gt_output_pars[0, :], label='Ground truth parameter c0')
        plt.plot(xTest, pred_output_pars, label='Predicted parameter c0')
        plt.axhline(0, color='black', lw=linew)
        plt.legend()
        plt.title("Predicted and ground truth output parameters")

    # Fig 5: Compactness
    plt.figure()
    plt.plot(xTrain, input_compactness, label='Input')
    plt.plot(xTrain, output_compactness, label='Output')
    plt.axhline(0, color='black', lw=linew)
    plt.grid()
    plt.legend()

    plt.show()
