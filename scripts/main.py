import argparse
import os
import sys
from shutil import copyfile

import numpy as np
import SimpleITK as sitk

from data.dicom_loader import DicomLoader

parser = argparse.ArgumentParser()
parser.add_argument('-surrogate_type', help='surrogate type: 0 - MR navigators/ 1 - ultrasound', type=int, required=True)
parser.add_argument('-project_root', help='root path of project directory', type=str, required=True)
parser.add_argument('-data_root', help='root path of data directory', type=str, required=True)
parser.add_argument('-data', help='subdirectory for data files', type=str, required=True)
parser.add_argument('-navi', help='subdirectory for navigators', type=str)
parser.add_argument('-us', help='subdirectory for us images', type=str)
parser.add_argument('-n_slices', help='number of slices per volume', type=int, required=True)
parser.add_argument('-n_sweeps', help='number of sweeps', type=int, required=True)
parser.add_argument('-n_training_sweeps', help='number of sweeps for GPR training', type=int, required=True)
parser.add_argument('-master_volume', help='master volume for 3D registration', type=str, required=True)
parser.add_argument('-input_format', help='input file format', type=str, default='png')
parser.add_argument('-output_format', help='output file format', type=str, default='vtk')
args = parser.parse_args()

if __name__ == "__main__":

    #### Preprocessing ####
    print('PREPROCESSING...')
    # Parse data files
    opt_data = args
    opt_data.is_navi = False
    opt_data.input_dir = os.path.join(opt_data.data_root, opt_data.data)
    opt_data.output_dir = os.path.join(opt_data.data_root, opt_data.data + "_mod")

    if not os.path.exists(opt_data.input_dir):
        sys.exit('Path to data files does not exist.')

    os.makedirs(opt_data.output_dir, exist_ok=True)

    data_loader = DicomLoader(opt_data)
    data_loader.preprocess()

    # Parse navi if required
    if args.surrogate_type == 0:
        opt_navi = args
        opt_navi.is_navi = True
        opt_navi.input_dir = os.path.join(opt_navi.data_root, opt_navi.navi)
        opt_navi.output_dir = os.path.join(opt_navi.data_root, opt_navi.navi + "_mod")

        if not os.path.exists(opt_navi.input_dir):
            sys.exit('Path to navigators does not exist.')

        os.makedirs(opt_navi.output_dir, exist_ok=True)

        navi_loader = DicomLoader(opt_navi)
        navi_loader.preprocess()

        # TODO: register navigators in 2D
        # /media/WDblue2/4DMRI/testCpp/zc_4dmri_ufssfp2_MK_Nav_F1_red/23_mod -o /media/WDblue2/4DMRI/testCpp/zc_4dmri_ufssfp2_MK_Nav_F1_red/reg_2d -rescale -wVerbose -optIter=150 -numScales 4 -regName=graph -regKernelSizeMax=4,6,10 -regKernelSizeMin=4,5,8 -regDecayType=linear -regDecayRate=-0.02 -optInitStep=0.8 -optName=AGD -optGrad -metric=lccA -metricKernelSizeMax=2,4,8 -metricKernelSizeMin=2,4,8 -edgeName=id -edgeScaleIntens=15 -edgeScaleDisplace=1 -edgeDecayDomain=0.00003 -edgeDetectNodeColapse -edgeNodeColapseMinVal=1 -edgeNodeColapseNewEdgeVal=1 -regStartEdgeUpdate=10 -regGraphMeanEdge -writeDICOM
    print('[done]')

    #### Stacking ####
    print('STACKING...')
    stacking_par_list = []
    if args.surrogate_type == 0:
        # TODO: check surrogate path
        stack_dir = os.path.join(args.data_root, 'stacks_navi')
        surrogate_dir = os.path.join(args.data_root, 'reg_2d/vtk/displacement_3.vtk')
        stacking_method = 'vonSiebenthal'
    elif args.surrogate_type == 1:
        stack_dir = os.path.join(args.data_root, 'stacks_us')
        surrogate_dir = os.path.join(args.data_root, 'us')
        stacking_method = 'ultrasound'
        stacking_par_list.append('-startUsIndex 0')
        stacking_par_list.append('-endUsIndex ' + str(args.n_sweeps * args.n_slices - 1))
    else:
        sys.exit('Surrogate not correctly defined')

    stacking_par_list.append('-surrogate ' + surrogate_dir)
    stacking_par_list.append('-o ' + stack_dir)
    stacking_par_list.append('-numberOfSweeps ' + str(args.n_sweeps))
    stacking_par_list.append('-numberOfSlicePos ' + str(args.n_slices))
    stacking_par_list.append('-dat ' + opt_data.output_dir)
    stacking_par_list.append('-stackingMethod ' + stacking_method)
    stacking_par_list.append('-save')

    stacking_pars = ' '.join(stacking_par_list)
    stacking_exe = os.path.join(args.project_root, 'ImageStacking4dMRI', 'build', 'ims4dMRI')
    stacking_cmd = ' '.join([stacking_exe, stacking_pars])

    os.makedirs(stack_dir, exist_ok=True)
    os.system(stacking_cmd)
    print('[done]')


    #### 3D Registration ####
    print('3D REGISTRATION...')
    registration3d_par_list = []
    if args.surrogate_type == 0:
        registration3d_dir = os.path.join(args.data_root, 'reg_3d_navi')
    elif args.surrogate_type == 1:
        registration3d_dir = os.path.join(args.data_root, 'reg_3d_us')
    else:
        sys.exit('Surrogate not correctly defined')

    registration3d_par_list.append('-r ' + os.path.join(stack_dir, args.master_volume))
    registration3d_par_list.append('-o ' + registration3d_dir)
    registration3d_par_list.append('-rescale')
    registration3d_par_list.append('-wVerbose')
    registration3d_par_list.append('-optIter=150')
    registration3d_par_list.append('-numScales 4')
    registration3d_par_list.append('-regName=graph')
    registration3d_par_list.append('-regKernelSizeMax=6,8,12')
    registration3d_par_list.append('-regKernelSizeMin=2,3,5')
    registration3d_par_list.append('-regDecayType=linear')
    registration3d_par_list.append('-regDecayRate=-0.02')
    registration3d_par_list.append('-optInitStep=0.8')
    registration3d_par_list.append('-optName=AGD')
    registration3d_par_list.append('-optGrad')
    registration3d_par_list.append('-metric=lccA')
    registration3d_par_list.append('-metricKernelSizeMax=4,6,10')
    registration3d_par_list.append('-metricKernelSizeMin=2,4,8')
    registration3d_par_list.append('-edgeName=id') #id #ne
    registration3d_par_list.append('-edgeScaleIntens=15')
    registration3d_par_list.append('-edgeScaleDisplace=1')
    registration3d_par_list.append('-edgeDecayDomain=0.003')
    registration3d_par_list.append('-edgeDetectNodeColapse')
    registration3d_par_list.append('-edgeNodeColapseMinVal=1')
    registration3d_par_list.append('-edgeNodeColapseNewEdgeVal=1')
    registration3d_par_list.append('-regStartEdgeUpdate=10')
    registration3d_par_list.append('-regGraphMeanEdge')
    registration3d_par_list.append('-verbose')
    registration3d_par_list.append('-cudaDeviceId 0')
    registration3d_par_list.append('-writeFlagFile')

    # Pairwise registration
    warped_dir = os.path.join(registration3d_dir, 'warpedImage')
    os.makedirs(warped_dir, exist_ok=True)

    dfs_dir = os.path.join(registration3d_dir, 'dfs')
    os.makedirs(dfs_dir, exist_ok=True)

    targets = sorted([os.path.join(stack_dir, i) for i in os.listdir(stack_dir) if i.startswith('vol')])
    print('Registration 3d: Number of target images: ' + str(len(targets)))

    registration3d_exe = os.path.join(args.project_root, 'GraphDemons_3D', 'build_3d', 'gdr')
    for itr, target in enumerate(targets):
        current_par_list = registration3d_par_list.copy()
        current_par_list.append('-t ' + target)

        registration3d_pars = ' '.join(current_par_list)
        registration3d_cmd = ' '.join([registration3d_exe, registration3d_pars])
        os.system(registration3d_cmd)

        # save registration result
        result_dir = os.path.join(registration3d_dir, "vtk")
        warped = sorted([os.path.join(result_dir, i) for i in os.listdir(result_dir) if i.startswith("warpedImage_")])
        warped_copy = os.path.join(warped_dir, ("%05d.vtk" % itr))
        copyfile(warped[-1], warped_copy)

        df = sorted([os.path.join(result_dir, i) for i in os.listdir(result_dir) if i.startswith("displacement_")])
        df_copy = os.path.join(dfs_dir, ("%05d.vtk" % itr))
        copyfile(df[-1], df_copy)
    print('[done]')


    #### Split data into training and test set ####
    print('SPLITTING...')
    n_imgs = args.n_sweeps*args.n_slices
    n_training_imgs = args.n_training_sweeps*args.n_slices
    n_test_imgs = n_imgs - n_training_imgs

    # Create directories
    sub_dir = {'surrogate': surrogate_dir,
               'dfs': dfs_dir,
               'warped': warped_dir}

    for name, current_dir in sub_dir.items():
        if current_dir == surrogate_dir:
            print(current_dir)
            format = args.input_format
        else:
            format = args.output_format

        files = sorted([os.path.join(current_dir, i) for i in os.listdir(current_dir) if i.endswith(format)])
        train_dir = os.path.join(current_dir, 'train')
        test_dir = os.path.join(current_dir, 'test')

        # Create or empty folder
        # train
        if os.path.isdir(train_dir):
            [os.remove(os.path.join(train_dir, f)) for f in os.listdir(train_dir)]
        else:
            os.makedirs(train_dir)

        # test
        if os.path.isdir(test_dir):
            [os.remove(os.path.join(test_dir, f)) for f in os.listdir(test_dir)]
        else:
            os.makedirs(test_dir)

        # copy all training files to train_dir
        for itr, file in enumerate(files[:n_training_imgs]):
            dest = os.path.join(train_dir, ('%05d.' % itr) + format)
            copyfile(file, dest)

        # copy all test files to train_dir
        for itr, file in enumerate(files[n_training_imgs:]):
            dest = os.path.join(test_dir, ('%05d.' % itr) + format)
            copyfile(file, dest)

        print('Splitting: Number of training images in' + train_dir +': ' + str(len(os.listdir(train_dir))))
        print('Splitting: Number of training images in' + test_dir +': ' + str(len(os.listdir(test_dir))))
    print('[done]')


    #### GP Regression ####
    print('GP REGRESSION...')
    gp_dir = os.path.join(registration3d_dir, 'gpr')
    gp_dir_pred = os.path.join(registration3d_dir, 'prediction')

    if not os.path.isdir(gp_dir):
        os.makedirs(gp_dir)
        gp_dir = os.path.join(gp_dir, 'gpr')

    if not os.path.isdir(gp_dir_pred):
        os.makedirs(gp_dir_pred)

    # learn
    kernel_string = "\"GaussianKernel(2.3, 1.0,)\""
    data_noise = "1.0"
    gp_learn_par_list = []
    gp_learn_par_list.append(os.path.join(surrogate_dir, 'train'))
    gp_learn_par_list.append(os.path.join(dfs_dir, 'train'))
    gp_learn_par_list.append(kernel_string)
    gp_learn_par_list.append(data_noise)
    gp_learn_par_list.append(gp_dir)
    print(gp_dir)

    gp_learn_pars = ' '.join(gp_learn_par_list)
    gp_learn_exe = os.path.join(args.project_root, 'GPR', 'build', 'apps', 'gpLearn')
    gp_learn_cmd = ' '.join([gp_learn_exe, gp_learn_pars])
    os.system(gp_learn_cmd)

    # predict
    print(gp_dir)
    gp_predict_par_list = []
    gp_predict_par_list.append(gp_dir)
    gp_predict_par_list.append(os.path.join(surrogate_dir, 'test'))
    gp_predict_par_list.append(gp_dir_pred)
    gp_predict_par_list.append(os.path.join(stack_dir, args.master_volume))

    gp_predict_pars = ' '.join(gp_predict_par_list)
    gp_predict_exe = os.path.join(args.project_root, 'GPR', 'build', 'apps', 'gpPredict')
    gp_predict_cmd = ' '.join([gp_predict_exe, gp_predict_pars])
    os.system(gp_predict_cmd)
    print('[done]')


    #### Evaluation ####
    print('EVALUATION...')
    result_dir = os.path.join(registration3d_dir, 'validation')
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    # Compute difference between ground-truth and gpr prediction
    dfs_test_dir = os.path.join(dfs_dir, 'test')
    dfs_true = sorted([os.path.join(dfs_test_dir, i) for i in os.listdir(dfs_test_dir) if i.endswith(args.output_format)])
    warped_test_dir = os.path.join(warped_dir, 'test')
    warped_true = sorted([os.path.join(warped_test_dir, i) for i in os.listdir(warped_test_dir)
                          if i.endswith(args.output_format)])
    stacks_true = sorted([os.path.join(stack_dir, i) for i in os.listdir(stack_dir) if i.startswith('vol')])

    dfs_pred = sorted([os.path.join(gp_dir_pred, i) for i in os.listdir(gp_dir_pred) if i.startswith('dfPred')])
    warped_pred = sorted([os.path.join(gp_dir_pred, i) for i in os.listdir(gp_dir_pred) if i.startswith('warpedImg')])

    # print(dfs_true)
    #print(dfs_pred)
    # print(warped_true)
    #print(warped_pred)
    #print(stacks_true)

    for itr in range(0, len(dfs_true)):
        # read images
        sitk_imgs = {
            'stack_true': sitk.ReadImage(stacks_true[itr]),
            'warped_true': sitk.ReadImage(warped_true[itr]),
            'warped_pred': sitk.ReadImage(warped_pred[itr]),
            'df_true': sitk.ReadImage(dfs_true[itr]),
            'df_pred': sitk.ReadImage(dfs_pred[itr])
        }

        # Option1: overwrite sitk images and save the information below
        # Option2: keep sitk images in dict, and automatically convert np to itk in a loop over the dict
        #origin = sitk_stack_true.GetOrigin()
        #spacing = sitk_stack_true.GetSpacing()
        #direction = sitk_stack_true.GetDirection()


        # Convert sitk to np
        np_imgs = {}
        for name, img in sitk_imgs.items():
            np_imgs[name] = sitk.GetArrayFromImage(img)

        # Qualitative comparison
        np_diff_stack = np.absolute(np_imgs['stack_true'] - np_imgs['warped_pred'])
        np_diff_warped = np.absolute(np_imgs['warped_true'] - np_imgs['warped_pred'])
        np_diff_df = np_imgs['df_true'] - np_imgs['df_pred']

        # loop?
        sitk_diff_stack = sitk.GetImageFromArray(np_diff_stack)
        sitk_diff_stack.SetDirection(sitk_imgs['stack_true'].GetDirection())
        sitk_diff_stack.SetSpacing(sitk_imgs['stack_true'].GetSpacing())
        sitk_diff_stack.SetOrigin(sitk_imgs['stack_true'].GetOrigin())

        sitk_diff_warped = sitk.GetImageFromArray(np_diff_warped)
        sitk_diff_warped.SetDirection(sitk_imgs['warped_true'].GetDirection())
        sitk_diff_warped.SetSpacing(sitk_imgs['warped_true'].GetSpacing())
        sitk_diff_warped.SetOrigin(sitk_imgs['warped_true'].GetOrigin())

        sitk_diff_df = sitk.GetImageFromArray(np_diff_df)
        sitk_diff_df.SetDirection(sitk_imgs['df_true'].GetDirection())
        sitk_diff_df.SetSpacing(sitk_imgs['df_true'].GetSpacing())
        sitk_diff_df.SetOrigin(sitk_imgs['df_true'].GetOrigin())

        # TODO: enumerate result!!!!
        sitk.WriteImage(sitk_diff_stack, os.path.join(result_dir, ('diff_stack%05d.vtk' % itr)))
        sitk.WriteImage(sitk_diff_stack, os.path.join(result_dir, ('diff_warped%05d.vtk' % itr)))
        sitk.WriteImage(sitk_diff_df, os.path.join(result_dir, ('diff_df%05d.vtk' % itr)))

    print('[done]')
