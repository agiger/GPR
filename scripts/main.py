import os
import sys
import shutil
from shutil import copyfile

import argparse
from argparse import Namespace
import yaml

import numpy as np
import SimpleITK as sitk

from data.dicom_loader import DicomLoader

parser = argparse.ArgumentParser()
parser.add_argument('-config', help='path to config.yaml file', type=str, default='./params/config.yaml')
args = parser.parse_args()

if __name__ == "__main__":
    # ----------------------------------------------------------
    # Load configuration parameters
    # ----------------------------------------------------------
    with open(args.config, 'r') as config_stream:
        cfg = yaml.safe_load(config_stream)

    opt = Namespace(**cfg['options'])
    cfg_general = Namespace(**cfg['general'])
    cfg_ref2d = cfg['reg2d']
    cfg_ref3d = cfg['reg3d']
    cfg_gpr = Namespace(**cfg['gpr'])

    # ----------------------------------------------------------
    # Preprocessing
    # ----------------------------------------------------------
    if opt.preprocess:
        print('PREPROCESSING...')
        # Parse data files
        opt_data = cfg_general
        opt_data.is_navi = False
        opt_data.input_dir = os.path.join(opt_data.data_root, opt_data.data_folder)
        opt_data.output_dir = os.path.join(opt_data.data_root, opt_data.data_folder + "_mod")

        if not os.path.exists(opt_data.input_dir):
            sys.exit('Path to data files does not exist.')

        if os.path.isdir(opt_data.output_dir) and opt.clear_data:
            if os.path.isdir(os.path.join(opt_data.output_dir, 'sorted')):
                shutil.rmtree(os.path.join(opt_data.output_dir, 'sorted'))
            [os.remove(os.path.join(opt_data.output_dir, f)) for f in os.listdir(opt_data.output_dir)]
        else:
            os.makedirs(opt_data.output_dir, exist_ok=True)

        data_loader = DicomLoader(opt_data)
        data_loader.preprocess()

        # Parse navi if required
        if cfg_general.surrogate_type == 0:
            opt_navi = cfg_general
            opt_navi.is_navi = True
            opt_navi.input_dir = os.path.join(opt_navi.data_root, opt_navi.navi_folder)
            opt_navi.output_dir = os.path.join(opt_navi.data_root, opt_navi.navi_folder + "_mod")

            if not os.path.exists(opt_navi.input_dir):
                sys.exit('Path to navigators does not exist.')

            if os.path.isdir(opt_navi.output_dir) and opt.clear_data:
                [os.remove(os.path.join(opt_navi.output_dir, f)) for f in os.listdir(opt_navi.output_dir)]
            else:
                os.makedirs(opt_navi.output_dir, exist_ok=True)

            navi_loader = DicomLoader(opt_navi)
            navi_loader.preprocess()

            if opt.perform_reg2d:
                do_something = 1
                # TODO: register navigators in 2D
                #  /media/WDblue2/4DMRI/testCpp/zc_4dmri_ufssfp2_MK_Nav_F1_red/23_mod -o /media/WDblue2/4DMRI/testCpp/zc_4dmri_ufssfp2_MK_Nav_F1_red/reg_2d -rescale -wVerbose -optIter=150 -numScales 4 -regName=graph -regKernelSizeMax=4,6,10 -regKernelSizeMin=4,5,8 -regDecayType=linear -regDecayRate=-0.02 -optInitStep=0.8 -optName=AGD -optGrad -metric=lccA -metricKernelSizeMax=2,4,8 -metricKernelSizeMin=2,4,8 -edgeName=id -edgeScaleIntens=15 -edgeScaleDisplace=1 -edgeDecayDomain=0.00003 -edgeDetectNodeColapse -edgeNodeColapseMinVal=1 -edgeNodeColapseNewEdgeVal=1 -regStartEdgeUpdate=10 -regGraphMeanEdge -writeDICOM

        print('[done]')

    # ----------------------------------------------------------
    # Stacking
    # ----------------------------------------------------------
    if opt.stack_4dmri:
        print('STACKING...')
        stacking_par_list = []
        if cfg_general.surrogate_type == 0:
            stack_dir = os.path.join(cfg_general.data_root, 'stacks_navi')
            surrogate_dir = os.path.join(cfg_general.data_root, 'reg_2d/vtk/displacement_3.vtk')
            stacking_method = 'vonSiebenthal'
        elif cfg_general.surrogate_type == 1:
            stack_dir = os.path.join(cfg_general.data_root, 'stacks_us')
            surrogate_dir = os.path.join(cfg_general.data_root, cfg_general.us_folder)
            stacking_method = 'ultrasound'
            stacking_par_list.append('-startUsIndex 0')
            stacking_par_list.append('-endUsIndex ' + str(cfg_general.n_sweeps * cfg_general.n_slices - 1))
        else:
            sys.exit('Surrogate not correctly defined')

        # Assert
        if not os.path.isdir(surrogate_dir):
            sys.exit('Path to surrogate data does not exist')

        stacking_par_list.append('-surrogate ' + surrogate_dir)
        stacking_par_list.append('-o ' + stack_dir)
        stacking_par_list.append('-numberOfSweeps ' + str(cfg_general.n_sweeps))
        stacking_par_list.append('-numberOfSlicePos ' + str(cfg_general.n_slices))
        stacking_par_list.append('-dat ' + opt_data.output_dir)
        stacking_par_list.append('-stackingMethod ' + stacking_method)
        stacking_par_list.append('-save')

        stacking_pars = ' '.join(stacking_par_list)
        stacking_exe = os.path.join(cfg_general.project_root, 'ImageStacking4dMRI', 'build', 'ims4dMRI')
        stacking_cmd = ' '.join([stacking_exe, stacking_pars])

        if os.path.isdir(stack_dir) and opt.clear_data:
            [os.remove(os.path.join(stack_dir, f)) for f in os.listdir(stack_dir)]
        else:
            os.makedirs(stack_dir, exist_ok=True)
        # os.system(stacking_cmd)
        print('[done]')

    # ----------------------------------------------------------
    # 3D Registration
    # ----------------------------------------------------------
    print('3D REGISTRATION...')
    if cfg_general.surrogate_type == 0:
        registration3d_dir = os.path.join(cfg_general.data_root, 'reg_3d_navi')
    elif cfg_general.surrogate_type == 1:
        registration3d_dir = os.path.join(cfg_general.data_root, 'reg_3d_us')
    else:
        sys.exit('Surrogate not correctly defined')

    registration3d_par_list = cfg_ref3d
    registration3d_par_list.append('-r ' + os.path.join(stack_dir, cfg_general.master_volume))
    registration3d_par_list.append('-o ' + registration3d_dir)

    # Pairwise registration
    warped_dir = os.path.join(registration3d_dir, 'warpedImage')
    if os.path.isdir(warped_dir) and opt.clear_data:
        [os.remove(os.path.join(warped_dir, f)) for f in os.listdir(warped_dir)]
    else:
        os.makedirs(warped_dir, exist_ok=True)

    dfs_dir = os.path.join(registration3d_dir, 'dfs')
    if os.path.isdir(dfs_dir) and opt.clear_data:
        [os.remove(os.path.join(dfs_dir, f)) for f in os.listdir(dfs_dir)]
    else:
        os.makedirs(dfs_dir, exist_ok=True)

    targets = sorted([os.path.join(stack_dir, i) for i in os.listdir(stack_dir) if i.startswith('vol')])
    print('Registration 3d: Number of target images: ' + str(len(targets)))

    registration3d_exe = os.path.join(cfg_general.project_root, 'GraphDemons_3D', 'build_3d', 'gdr')
    for itr, target in enumerate(targets):
        current_par_list = registration3d_par_list.copy()
        current_par_list.append('-t ' + target)

        registration3d_pars = ' '.join(current_par_list)
        registration3d_cmd = ' '.join([registration3d_exe, registration3d_pars])
        # os.system(registration3d_cmd)

        # save registration result
        result_dir = os.path.join(registration3d_dir, "vtk")
        warped = sorted([os.path.join(result_dir, i) for i in os.listdir(result_dir) if i.startswith("warpedImage_")])
        warped_copy = os.path.join(warped_dir, ("warpedImg%05d.vtk" % itr))
        # copyfile(warped[-1], warped_copy)

        df = sorted([os.path.join(result_dir, i) for i in os.listdir(result_dir) if i.startswith("displacement_")])
        df_copy = os.path.join(dfs_dir, ("dfReg%05d.vtk" % itr))
        # copyfile(df[-1], df_copy)
    print('[done]')

    # ----------------------------------------------------------
    # Split data into training and test set
    # ----------------------------------------------------------
    print('SPLITTING...')
    n_imgs = cfg_general.n_sweeps*cfg_general.n_slices
    n_training_imgs = cfg_general.n_training_sweeps*cfg_general.n_slices
    n_test_imgs = n_imgs - n_training_imgs

    # Create directories
    sub_dir = {'surrogate': surrogate_dir,
               'dfs': dfs_dir,
               'warped': warped_dir}

    for name, current_dir in sub_dir.items():
        if current_dir == surrogate_dir:
            format = cfg_general.input_format
        else:
            format = cfg_general.output_format

        files = sorted([os.path.join(current_dir, i) for i in os.listdir(current_dir) if i.endswith(format)])
        train_dir = os.path.join(current_dir, 'train')
        test_dir = os.path.join(current_dir, 'test')

        # Create or empty folder
        # train
        if os.path.isdir(train_dir) and opt.clear_data:
            [os.remove(os.path.join(train_dir, f)) for f in os.listdir(train_dir)]
        else:
            os.makedirs(train_dir, exist_ok=True)

        # test
        if os.path.isdir(test_dir) and opt.clear_data:
            [os.remove(os.path.join(test_dir, f)) for f in os.listdir(test_dir)]
        else:
            os.makedirs(test_dir, exist_ok=True)

        # copy all training files to train_dir
        for itr, file in enumerate(files[:n_training_imgs]):
            dest = os.path.join(train_dir, ('%05d.' % itr) + format)
            # copyfile(file, dest)

        # copy all test files to train_dir
        for itr, file in enumerate(files[n_training_imgs:]):
            dest = os.path.join(test_dir, ('%05d.' % itr) + format)
            # copyfile(file, dest)

        print('Splitting: Number of training images in' + train_dir +': ' + str(len(os.listdir(train_dir))))
        print('Splitting: Number of training images in' + test_dir +': ' + str(len(os.listdir(test_dir))))
    print('[done]')

    # ----------------------------------------------------------
    # GP Regression
    # ----------------------------------------------------------
    print('GP REGRESSION...')
    gp_dir = os.path.join(registration3d_dir, 'gpr')
    if os.path.isdir(gp_dir) and opt.clear_data:
        [os.remove(os.path.join(gp_dir, f)) for f in os.listdir(gp_dir)]
    else:
        os.makedirs(gp_dir, exist_ok=True)
    gp_dir = os.path.join(gp_dir, 'gpr')

    gp_dir_pred = os.path.join(registration3d_dir, 'prediction')
    if os.path.isdir(gp_dir_pred) and opt.clear_data:
        [os.remove(os.path.join(gp_dir_pred, f)) for f in os.listdir(gp_dir_pred)]
    else:
        os.makedirs(gp_dir_pred, exist_ok=True)

    # Learn
    gp_learn_par_list = []
    gp_learn_par_list.append(os.path.join(surrogate_dir, 'train'))
    gp_learn_par_list.append(os.path.join(dfs_dir, 'train'))
    gp_learn_par_list.append(cfg_gpr.kernel_string)
    gp_learn_par_list.append(str(cfg_gpr.data_noise))
    gp_learn_par_list.append(gp_dir)

    gp_learn_pars = ' '.join(gp_learn_par_list)
    gp_learn_exe = os.path.join(cfg_general.project_root, 'GPR', 'build', 'apps', 'gpLearn')
    gp_learn_cmd = ' '.join([gp_learn_exe, gp_learn_pars])
    # os.system(gp_learn_cmd)

    # Predict
    gp_predict_par_list = []
    gp_predict_par_list.append(gp_dir)
    # gp_predict_par_list.append(os.path.join(surrogate_dir, 'test'))
    gp_predict_par_list.append(os.path.join(surrogate_dir, 'train'))
    gp_predict_par_list.append(gp_dir_pred)
    gp_predict_par_list.append(os.path.join(stack_dir, cfg_general.master_volume))

    gp_predict_pars = ' '.join(gp_predict_par_list)
    gp_predict_exe = os.path.join(cfg_general.project_root, 'GPR', 'build', 'apps', 'gpPredict')
    gp_predict_cmd = ' '.join([gp_predict_exe, gp_predict_pars])
    # os.system(gp_predict_cmd)
    print('[done]')

    # ----------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------
    print('EVALUATION...')
    result_dir = os.path.join(registration3d_dir, 'validation')
    if os.path.isdir(result_dir) and opt.clear_data:
        [os.remove(os.path.join(result_dir, f)) for f in os.listdir(result_dir)]
    else:
        os.makedirs(result_dir, exist_ok=True)

    # Compute difference between ground-truth and gpr prediction
    # dfs_test_dir = os.path.join(dfs_dir, 'test')
    dfs_test_dir = os.path.join(dfs_dir, 'train')
    dfs_true = sorted([os.path.join(dfs_test_dir, i) for i in os.listdir(dfs_test_dir)
                       if i.endswith(cfg_general.output_format)])
    # warped_test_dir = os.path.join(warped_dir, 'test')
    warped_test_dir = os.path.join(warped_dir, 'train')
    warped_true = sorted([os.path.join(warped_test_dir, i) for i in os.listdir(warped_test_dir)
                          if i.endswith(cfg_general.output_format)])
    stacks_true = sorted([os.path.join(stack_dir, i) for i in os.listdir(stack_dir) if i.startswith('vol')])

    dfs_pred = sorted([os.path.join(gp_dir_pred, i) for i in os.listdir(gp_dir_pred) if i.startswith('dfPred')])
    warped_pred = sorted([os.path.join(gp_dir_pred, i) for i in os.listdir(gp_dir_pred) if i.startswith('warpedImg')])

    for itr in range(0, len(dfs_true)):
        # read images
        sitk_imgs = {
            'stack_true': sitk.ReadImage(stacks_true[itr]),
            'warped_true': sitk.ReadImage(warped_true[itr]),
            'warped_pred': sitk.ReadImage(warped_pred[itr]),
            'df_true': sitk.ReadImage(dfs_true[itr]),
            'df_pred': sitk.ReadImage(dfs_pred[itr])
        }

        # Convert sitk to np
        np_imgs = {}
        for name, img in sitk_imgs.items():
            np_imgs[name] = sitk.GetArrayFromImage(img)

        # Qualitative comparison
        np_diff = {
            'stack': np.absolute(np_imgs['stack_true'] - np_imgs['warped_pred']),
            'warped': np.absolute(np_imgs['warped_true']*4095 - np_imgs['warped_pred']),
            'df': np_imgs['df_true'] - np_imgs['df_pred']
        }

        sitk_diff = {}
        for name, img in np_diff.items():
            diff = sitk.GetImageFromArray(img)
            diff.SetDirection(sitk_imgs[name + '_true'].GetDirection())
            diff.SetSpacing(sitk_imgs[name + '_true'].GetSpacing())
            diff.SetOrigin(sitk_imgs[name + '_true'].GetOrigin())

            sitk_diff[name] = diff
            # sitk.WriteImage(diff, os.path.join(result_dir, ('diff_' + name + '%05d.vtk' % itr)))

    print('[done]')
