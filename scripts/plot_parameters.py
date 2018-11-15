import argparse

import csv
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-input_par', help='input parameter file (csv)', type=str)
parser.add_argument('-output_par', help='output parameter file (csv)', type=str)
parser.add_argument('-pred_input_par', help='predicted input parameter file (csv)', type=str)
parser.add_argument('-pred_output_par', help='predicted output parameter file (csv)', type=str)
args = parser.parse_args()

if __name__ == "__main__":
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


    # Plot
    x1 = list(range(0, len(input_pars)))
    y1 = list(map(list, zip(*input_pars)))
    fig1, axs1 = plt.subplots(nrows=len(y1), ncols=1)
    fig1.suptitle("Input parameters for training")
    axs1 = axs1.ravel()
    for i in range(0, len(y1)):
        y1[i] = list(map(float, y1[i]))
        axs1[i].plot(x1, y1[i], label='Parameter c{}'.format(i + 1))
        axs1[i].legend()

    x2 = list(range(0, len(output_pars)))
    y2 = list(map(list, zip(*output_pars)))
    fig2, axs2 = plt.subplots(nrows=len(y2), ncols=1)
    fig2.suptitle("Output parameters for training")
    axs2 = axs2.ravel()
    for i in range(0, len(y2)):
        y2[i] = list(map(float, y2[i]))
        axs2[i].plot(x2, y2[i], label='Parameter c{}'.format(i + 1))
        axs2[i].legend()

    x3 = list(range(0, len(pred_input_pars)))
    y3 = list(map(list, zip(*pred_input_pars)))
    fig3, axs3 = plt.subplots(nrows=len(y3), ncols=1)
    fig3.suptitle("Input parameters for prediction")
    axs3 = axs3.ravel()
    for i in range(0, len(y3)):
        y3[i] = list(map(float, y3[i]))
        axs3[i].plot(x3, y3[i], label='Parameter c{}'.format(i + 1))
        axs3[i].legend()

    x4 = list(range(0, len(pred_output_pars[0])))
    y4 = pred_output_pars
    # y4 = list(map(list, zip(*pred_output_pars)))
    fig4, axs4 = plt.subplots(nrows=len(y4), ncols=1)
    fig4.suptitle("Predicted output parameters")
    axs4 = axs4.ravel()
    for i in range(0, len(y4)):
        y4[i] = list(map(float, y4[i]))
        axs4[i].plot(x4, y4[i], label='Parameter c{}'.format(i + 1))
        axs4[i].legend()
    plt.show()
