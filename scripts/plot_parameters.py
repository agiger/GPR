import argparse
import os

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib2tikz

parser = argparse.ArgumentParser()
parser.add_argument('-root', help='root directory of parameter files', type=str, required=True)
parser.add_argument('-input_par', help='input parameter file (csv)', type=str)
parser.add_argument('-output_par', help='output parameter file (csv)', type=str)
parser.add_argument('-pred_input_par', help='predicted input parameter file (csv)', type=str)
parser.add_argument('-pred_output_par', help='predicted output parameter file (csv)', type=str)
parser.add_argument('-gt_output_par', help='ground truth output parameter file (csv)', type=str)
parser.add_argument('-input_compactness', help='input compactness file (csv)', type=str)
parser.add_argument('-output_compactness', help='output compactness file (csv)', type=str)
parser.add_argument('-dest', help='output folder for saving plots', type=str)
args = parser.parse_args()

if __name__ == "__main__":
    plot_nInputC = 2
    plot_nOutputC = 2
    plot_fig = [1, 1, 0, 0, 0, 1, 1]

    # input parameters
    input_pars = np.loadtxt(os.path.join(args.root, args.input_par), delimiter=',')
    print('input_pars', input_pars.shape)

    # output parameters
    output_pars = np.loadtxt(os.path.join(args.root, args.output_par), delimiter=',')
    print('output_pars', output_pars.shape)

    # input parameters for prediction
    pred_input_pars = np.loadtxt(os.path.join(args.root, args.pred_input_par), delimiter=',')
    print('pred_input_pars', pred_input_pars.shape)

    # predicted output parameters
    pred_output_pars = np.loadtxt(os.path.join(args.root, args.pred_output_par), delimiter=',')
    print('pred_output_pars', pred_output_pars.shape)

    # ground truth output parameters
    gt_output_pars = np.loadtxt(os.path.join(args.root, args.gt_output_par), delimiter=',')
    print('gt_output_pars', gt_output_pars.shape)

    # input compactness
    input_compactness = np.loadtxt(os.path.join(args.root, args.input_compactness), delimiter=',')
    print('input_compactness', input_compactness.shape)

    # output compactness
    output_compactness = np.loadtxt(os.path.join(args.root, args.output_compactness), delimiter=',')
    print('output_compactness', output_compactness.shape)

    # Plot
    itr = 0
    linew = 0.5
    offset = 13
    nTrainFiles = input_pars.shape[1]
    nTestFiles = pred_input_pars.shape[1]
    xTrain = np.arange(offset, nTrainFiles+offset)
    xTest = np.arange(0, nTestFiles)

    n_plot_samples = 100
    # Fig 1: Input parameters for training
    if plot_fig[itr]:
        if plot_nInputC > 1:
            fig1, axs1 = plt.subplots(nrows=plot_nInputC, ncols=1)
            fig1.suptitle("Input parameters for training")
            axs1 = axs1.ravel()
            for i in range(0, plot_nInputC):
                axs1[i].plot(xTrain[0:n_plot_samples], input_pars[i, 0:n_plot_samples], label='Parameter c{}'.format(i))
                axs1[i].axhline(0, color='black', lw=linew)
                axs1[i].set_xlabel('time points')
                axs1[i].set_ylabel('PC_{}'.format(i))
                axs1[i].set_yticks([-1, 0, 1])
                # axs1[i].legend()
                axs1[i].grid()
        else:
            fig1 = plt.figure()
            plt.plot(xTrain[0:n_plot_samples], input_pars[0, 0:n_plot_samples], label='Parameter c0')
            plt.axhline(0, color='black', lw=linew)
            plt.xlabel('time points')
            plt.ylabel('PC_0')
            plt.grid()
            plt.legend()
            plt.title("Input parameters for training")
        fig1.savefig(os.path.join(args.dest, 'input_par_for_training.pdf'), bbox_inches='tight')
        matplotlib2tikz.save(os.path.join(args.dest, "input_par_for_training.tex"))

    # Fig 2: Output parameters for training
    itr += 1
    if plot_fig[itr]:
        if plot_nOutputC > 1:
            fig2, axs2 = plt.subplots(nrows=plot_nOutputC, ncols=1)
            fig2.suptitle("Output parameters for training")
            axs2 = axs2.ravel()
            for i in range(0, plot_nOutputC):
                axs2[i].plot(xTrain[0:n_plot_samples], -output_pars[i, 0:n_plot_samples], label='Parameter c{}'.format(i))
                axs2[i].axhline(0, color='black', lw=linew)
                axs2[i].set_xlabel('time points')
                axs2[i].set_ylabel('PC_{}'.format(i))
                axs2[i].set_yticks([-1, 0, 1])
                # axs2[i].legend()
                axs2[i].grid()
        else:
            fig2 = plt.figure()
            plt.plot(xTrain[0:n_plot_samples], output_pars[0, 0:n_plot_samples], label='Parameter c0')
            plt.axhline(0, color='black', lw=linew)
            plt.xlabel('time points')
            plt.ylabel('PC_0')
            plt.grid()
            plt.yticks([-1, 0, 1])
            plt.legend()
            plt.title("Output parameters for training")
        fig2.savefig(os.path.join(args.dest, 'output_par_for_training.pdf'), bbox_inches='tight')

    # Fig 3: Input parameters for prediction
    itr += 1
    if plot_fig[itr]:
        if plot_nInputC > 1:
            fig3, axs3 = plt.subplots(nrows=plot_nInputC, ncols=1)
            fig3.suptitle("Input parameters for prediction")
            axs3 = axs3.ravel()
            for i in range(0, plot_nInputC):
                axs3[i].plot(xTest, pred_input_pars[i, :], label='Parameter c{}'.format(i))
                axs3[i].axhline(0, color='black', lw=linew)
                axs3[i].legend()
        else:
            fig3 = plt.figure()
            plt.plot(xTest, pred_input_pars[0, :], label='Parameter c0')
            plt.axhline(0, color='black', lw=linew)
            plt.legend()
            plt.title("Input parameters for prediction")
        fig3.savefig(os.path.join(args.dest, 'input_par_for_prediction.pdf'), bbox_inches='tight')

    # Fig 4: Predicted output parameters
    itr += 1
    if plot_fig[itr]:
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
            fig4 = plt.figure()
            plt.plot(xTest, gt_output_pars[0, :], label='Ground truth parameter c0')
            plt.plot(xTest, pred_output_pars, label='Predicted parameter c0')
            plt.axhline(0, color='black', lw=linew)
            plt.legend()
            plt.title("Predicted and ground truth output parameters")
        fig4.savefig(os.path.join(args.dest, 'output_par_for_prediction.pdf'), bbox_inches='tight')

    # Fig 5: Compactness
    itr += 1
    if plot_fig[itr]:
        fig5 = plt.figure()
        plt.plot(xTrain, input_compactness, label='Input')
        plt.plot(xTrain, output_compactness, label='Output')
        plt.axhline(0, color='black', lw=linew)
        plt.grid()
        plt.legend()
        fig5.savefig(os.path.join(args.dest, 'compactness.pdf'), bbox_inches='tight')

    # Fig 6: Input vs Output c0
    itr += 1
    if plot_fig[itr]:
        fig6 = plt.figure()
        plt.plot(xTrain[0:n_plot_samples], input_pars[0, 0:n_plot_samples], label='Input c0')
        plt.plot(xTrain[0:n_plot_samples], -output_pars[0, 0:n_plot_samples], label='Output c0')
        plt.axhline(0, color='black', lw=linew)
        plt.grid()
        plt.legend()
        fig6.savefig(os.path.join(args.dest, 'input_output_c0.pdf'), bbox_inches='tight')

    # Fig 7: Input parameters for training
    itr += 1
    if plot_fig[itr]:
        fig7 = plt.figure()
        plt.scatter(input_pars[0, :], input_pars[1, :])
        plt.xlabel('PC_0')
        plt.ylabel('PC_1')
        #plt.axhline(0, color='black', lw=linew)
        plt.grid()
        plt.title("Prinicipal components")
        fig7.savefig(os.path.join(args.dest, 'input_pcs.pdf'), bbox_inches='tight')
        matplotlib2tikz.save(os.path.join(args.dest, "input_pcs.tex"))

    plt.show()
