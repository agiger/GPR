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
parser.add_argument('--config', help='path to config.yaml file', type=str, default='./params/config.yaml')
args = parser.parse_args()


def pairwise_registration(dirs, par_list, master, refs, exe):
    reg_dir = dirs['registration_dir']
    # warped_dir = dirs['warped_dir']
    dfs_dir = dirs['dfs_dir']

    par_list.append('-t ' + master)
    par_list.append('-o ' + reg_dir)

    # Pairwise registration
    # if os.path.isdir(warped_dir):
    #     shutil.rmtree(warped_dir)
    # os.makedirs(warped_dir, exist_ok=True)

    if os.path.isdir(dfs_dir):
        shutil.rmtree(dfs_dir)
    os.makedirs(dfs_dir, exist_ok=True)

    for itr, ref in enumerate(refs):
        current_par_list = par_list.copy()
        current_par_list.append('-r ' + ref)

        reg_pars = ' '.join(current_par_list)
        reg_cmd = ' '.join([exe, reg_pars])
        os.system(reg_cmd)

        # save registration result
        result_dir = os.path.join(reg_dir, "vtk")
        # warped = sorted([os.path.join(result_dir, i) for i in os.listdir(result_dir) if i.startswith("warpedImage_")])
        # warped_copy = os.path.join(warped_dir, ("warpedImg%05d.vtk" % itr))
        # copyfile(warped[-1], warped_copy)

        df = sorted([os.path.join(result_dir, i) for i in os.listdir(result_dir) if i.startswith("displacement_")])
        df_copy = os.path.join(dfs_dir, ("dfReg%05d.vtk" % itr))
        copyfile(df[-1], df_copy)


if __name__ == "__main__":
    # ----------------------------------------------------------
    # Load configuration parameters
    # ----------------------------------------------------------
    with open(args.config, 'r') as config_stream:
        cfg = yaml.safe_load(config_stream)

    opt = Namespace(**cfg['options'])
    exe = Namespace(**cfg['exe'])
    cfg_general = Namespace(**cfg['general'])
    cfg_reg2d = cfg['reg2d']
    cfg_reg3d = cfg['reg3d']
    cfg_gpr = Namespace(**cfg['gpr'])

    # ----------------------------------------------------------
    # Preprocessing
    # ----------------------------------------------------------
    # Parse data files
    opt_data = Namespace(**cfg['general'])
    opt_data.is_navi = False
    opt_data.input_dir = os.path.join(opt_data.root_dir, opt_data.data_dir)
    opt_data.output_dir = os.path.join(opt_data.root_dir, opt_data.data_dir + "_mod")

    if opt.preprocessing:
        print('PREPROCESSING DATA FILES...')
        # Assert
        if not os.path.exists(opt_data.input_dir):
            sys.exit('Path to data files does not exist.')

        if os.path.isdir(opt_data.output_dir):
            if os.path.isdir(os.path.join(opt_data.output_dir, 'sorted')):
                shutil.rmtree(os.path.join(opt_data.output_dir, 'sorted'))
            [os.remove(os.path.join(opt_data.output_dir, f)) for f in os.listdir(opt_data.output_dir)]
        else:
            os.makedirs(opt_data.output_dir, exist_ok=True)

        data_loader = DicomLoader(opt_data)
        data_loader.preprocess()
        print('[done]')

    # Parse navi if required
    if cfg_general.surrogate_type == 0:
        opt_navi = Namespace(**cfg['general'])
        opt_navi.is_navi = True
        opt_navi.input_dir = os.path.join(opt_navi.root_dir, opt_navi.navi_dir)
        opt_navi.output_dir = os.path.join(opt_navi.root_dir, opt_navi.navi_dir + "_mod")

        if opt.preprocessing:
            print('PREPROCESSING NAVIS...')
            if not os.path.exists(opt_navi.input_dir):
                sys.exit('Path to navigators does not exist.')

            if os.path.isdir(opt_navi.output_dir):
                [os.remove(os.path.join(opt_navi.output_dir, f)) for f in os.listdir(opt_navi.output_dir)]
            else:
                os.makedirs(opt_navi.output_dir, exist_ok=True)

            navi_loader = DicomLoader(opt_navi)
            navi_loader.preprocess()
            print('[done]')

        registration2d_dir = os.path.join(cfg_general.root_dir, 'reg_2d')
        reg2d_dirs = {
            'registration_dir': registration2d_dir,
            'warped_dir': os.path.join(registration2d_dir, 'warpedImage'),
            'dfs_dir': os.path.join(registration2d_dir, 'dfs')
        }

        if opt.registration_2d:
            print('2D REGISTRATION OF NAVIS')
            refs = sorted([os.path.join(opt_navi.output_dir, i) for i in os.listdir(opt_navi.output_dir) if i.startswith('navi')])
            target = os.path.join(opt_navi.output_dir, cfg_general.master_navi)
            print('Registration 2d: Number of reference images: ' + str(len(refs)))

            pairwise_registration(reg2d_dirs, cfg_reg2d, target, refs, exe.registration_2d)
            print('[done]')

    # ----------------------------------------------------------
    # Stacking
    # ----------------------------------------------------------
    # TODO: change order of assignment (if input_dir defined, use this path always)
    stacking_par_list = []
    if cfg_general.surrogate_type == 0:
        stack_dir = os.path.join(cfg_general.root_dir, 'stacks_navi')
        surrogate_dir = os.path.join(registration2d_dir, 'dfs')
        stacking_method = 'vonSiebenthal'
        series_format = 'dfReg%05d.vtk'
    elif cfg_general.surrogate_type == 1:
        stack_dir = os.path.join(cfg_general.root_dir, 'stacks_us')
        surrogate_dir = os.path.join(cfg_general.root_dir, cfg_general.us_dir)
        stacking_method = 'ultrasound'
        series_format = '%05d.png'
    else:
        try:
            surrogate_dir = os.path.join(cfg_general.root_dir, cfg_general.input_dir)
            print(surrogate_dir)
        except:
            sys.exit('Surrogate not correctly defined')

    # Assert
    if not os.path.isdir(surrogate_dir):
        sys.exit('Path to surrogate data does not exist')

    if opt.stacking:
        print('STACKING...')
        stacking_par_list.append('-o ' + stack_dir)
        stacking_par_list.append('-data ' + opt_data.output_dir)
        stacking_par_list.append('-surrogate ' + surrogate_dir)
        stacking_par_list.append('-startIndex 0')
        stacking_par_list.append('-endIndex ' + str(cfg_general.n_sweeps * cfg_general.n_slices - 1))
        stacking_par_list.append('-seriesFormat ' + series_format)
        stacking_par_list.append('-numberOfSweeps ' + str(cfg_general.n_sweeps))
        stacking_par_list.append('-numberOfSlicePos ' + str(cfg_general.n_slices))
        stacking_par_list.append('-stackingMethod ' + stacking_method)
        stacking_par_list.append('-save')

        stacking_pars = ' '.join(stacking_par_list)
        stacking_cmd = ' '.join([exe.stacking, stacking_pars])

        if os.path.isdir(stack_dir):
            [os.remove(os.path.join(stack_dir, f)) for f in os.listdir(stack_dir)]
        else:
            os.makedirs(stack_dir, exist_ok=True)

        os.system(stacking_cmd)
        print('[done]')

    # ----------------------------------------------------------
    # 3D Registration
    # ----------------------------------------------------------
    # TODO: change order of assignment (if output_dir defined, use this path always)
    if cfg_general.surrogate_type == 0:
        registration3d_dir = os.path.join(cfg_general.root_dir, 'reg_3d_navi')
    elif cfg_general.surrogate_type == 1:
        registration3d_dir = os.path.join(cfg_general.root_dir, 'reg_3d_us')
    else:
        try:
            registration3d_dir = os.path.join(cfg_general.root_dir, cfg_general.output_dir)
        except:
            sys.exit('Data directory not correctly defined')

    reg3d_dirs = {
        'registration_dir': registration3d_dir,
        'warped_dir': os.path.join(registration3d_dir, 'warpedImage'),
        'dfs_dir': os.path.join(registration3d_dir, 'dfs')
    }

    if opt.registration_3d:
        print('3D REGISTRATION...')
        refs = sorted([os.path.join(stack_dir, i) for i in os.listdir(stack_dir) if i.startswith('vol')])
        target = os.path.join(stack_dir, cfg_general.master_volume)
        print('Registration 3d: Number of reference images: ' + str(len(refs)))

        pairwise_registration(reg3d_dirs, cfg_reg3d, target, refs, exe.registration_3d)
        print('[done]')

    # ----------------------------------------------------------
    # Split data into training and test set
    # ----------------------------------------------------------
    if opt.splitting_data or ((opt.registration_2d or opt.registration_3d) and opt.regression):
        print('SPLITTING...')
        n_imgs = cfg_general.n_sweeps*cfg_general.n_slices
        n_training_imgs = cfg_general.n_training_sweeps*cfg_general.n_slices
        n_test_imgs = n_imgs - n_training_imgs

        # Create directories
        sub_dir = {'surrogate': surrogate_dir,
                   'dfs': reg3d_dirs['dfs_dir'],
                   'warped': reg3d_dirs['warped_dir']}

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
            if os.path.isdir(train_dir):
                [os.remove(os.path.join(train_dir, f)) for f in os.listdir(train_dir)]
            else:
                os.makedirs(train_dir, exist_ok=True)

            # test
            if os.path.isdir(test_dir):
                [os.remove(os.path.join(test_dir, f)) for f in os.listdir(test_dir)]
            else:
                os.makedirs(test_dir, exist_ok=True)

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

    # ----------------------------------------------------------
    # GP Regression
    # ----------------------------------------------------------
    subdir = cfg_gpr.subdir  # validation, test, (train)
    gp_dir_pred = os.path.join(registration3d_dir, '{:s}_pred'.format(subdir))

    if opt.regression:
        print('GP REGRESSION...')
        gp_dir = os.path.join(registration3d_dir, 'gpr')
        if os.path.isdir(gp_dir):
            if not cfg_gpr.use_precomputed:
                [os.remove(os.path.join(gp_dir, f)) for f in os.listdir(gp_dir)]
        else:
            os.makedirs(gp_dir, exist_ok=True)
        gp_dir = os.path.join(gp_dir, 'gpr')

        if os.path.isdir(gp_dir_pred):
            [os.remove(os.path.join(gp_dir_pred, f)) for f in os.listdir(gp_dir_pred)]
        else:
            os.makedirs(gp_dir_pred, exist_ok=True)

        # Learn
        gp_learn_par_list = []
        gp_learn_par_list.append(os.path.join(surrogate_dir, 'train'))
        # gp_learn_par_list.append(os.path.join(reg3d_dirs['dfs_dir'], 'train'))
        gp_learn_par_list.append(os.path.join(registration3d_dir, 'train'))
        gp_learn_par_list.append(cfg_gpr.kernel_string)
        gp_learn_par_list.append(str(cfg_gpr.data_noise))
        gp_learn_par_list.append(gp_dir)
        gp_learn_par_list.append(cfg_gpr.n_inputModes)
        gp_learn_par_list.append(cfg_gpr.n_outputModes)
        if cfg_gpr.use_precomputed:
            gp_learn_par_list.append('true')

        gp_learn_pars = ' '.join(gp_learn_par_list)
        gp_learn_cmd = ' '.join([exe.regression_learn, gp_learn_pars])
        os.system(gp_learn_cmd)

        # Predict
        gp_predict_par_list = []
        gp_predict_par_list.append(gp_dir)
        gp_predict_par_list.append(os.path.join(surrogate_dir, subdir))
        # gp_predict_par_list.append(os.path.join(surrogate_dir, 'train'))
        gp_predict_par_list.append(gp_dir_pred)
        gp_predict_par_list.append(os.path.join(registration3d_dir, subdir))
        # gp_predict_par_list.append(os.path.join(stack_dir, cfg_general.master_volume))
        gp_predict_par_list.append(os.path.join(cfg_general.root_dir, cfg_general.master_volume))
        gp_predict_par_list.append(cfg_gpr.n_inputModes)
        gp_predict_par_list.append(cfg_gpr.n_outputModes)
        if subdir == 'test':
            gp_predict_par_list.append('false')  # use_precomputed
            gp_predict_par_list.append('test')
        elif cfg_gpr.use_precomputed_features:
            gp_predict_par_list.append('true')  # use_precomputed_features

        gp_predict_pars = ' '.join(gp_predict_par_list)
        gp_predict_cmd = ' '.join([exe.regression_predict, gp_predict_pars])
        os.system(gp_predict_cmd)
        print('[done]')

    # ----------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------
    result_dir = os.path.join(registration3d_dir, '{:s}_diff'.format(subdir))
    if opt.evaluation:
        print('EVALUATION...')
        if os.path.isdir(result_dir):
            [os.remove(os.path.join(result_dir, f)) for f in os.listdir(result_dir)]
        else:
            os.makedirs(result_dir, exist_ok=True)

        # Compute difference between ground-truth and gpr prediction
        # dfs_test_dir = os.path.join(reg3d_dirs['dfs_dir'], subdir)
        dfs_test_dir = os.path.join(registration3d_dir, subdir)
        # dfs_test_dir = os.path.join(reg3d_dirs['dfs_dir'], 'train')
        dfs_true = sorted([os.path.join(dfs_test_dir, i) for i in os.listdir(dfs_test_dir)
                           if i.endswith(cfg_general.output_format)])
        # warped_test_dir = os.path.join(reg3d_dirs['warped_dir'], 'test')
        # warped_test_dir = os.path.join(reg3d_dirs['warped_dir'], 'train')
        # warped_true = sorted([os.path.join(warped_test_dir, i) for i in os.listdir(warped_test_dir)
        #                       if i.endswith(cfg_general.output_format)])
        # stacks_true = sorted([os.path.join(stack_dir, i) for i in os.listdir(stack_dir) if i.startswith('vol')])

        dfs_pred = sorted([os.path.join(gp_dir_pred, i) for i in os.listdir(gp_dir_pred) if i.startswith('dfPred')])
        # warped_pred = sorted([os.path.join(gp_dir_pred, i) for i in os.listdir(gp_dir_pred) if i.startswith('warpedImg')])

        for itr in range(0, len(dfs_true)):
            # read images
            sitk_imgs = {
                # 'stack_true': sitk.ReadImage(stacks_true[itr]),
                # 'warped_true': sitk.ReadImage(warped_true[itr]),
                # 'warped_pred': sitk.ReadImage(warped_pred[itr]),
                'df_true': sitk.ReadImage(dfs_true[itr]),
                'df_pred': sitk.ReadImage(dfs_pred[itr])
            }

            # Convert sitk to np
            np_imgs = {}
            for name, img in sitk_imgs.items():
                np_imgs[name] = sitk.GetArrayFromImage(img)

            # Qualitative comparison
            np_diff = {
                # 'stack': np.absolute(np_imgs['stack_true'] - np_imgs['warped_pred']),
                # 'warped': np.absolute(np_imgs['warped_true']*4095 - np_imgs['warped_pred']),
                'df': np_imgs['df_true'] - np_imgs['df_pred']
            }

            sitk_diff = {}
            for name, img in np_diff.items():
                diff = sitk.GetImageFromArray(img)
                diff.SetDirection(sitk_imgs[name + '_true'].GetDirection())
                diff.SetSpacing(sitk_imgs[name + '_true'].GetSpacing())
                diff.SetOrigin(sitk_imgs[name + '_true'].GetOrigin())

                sitk_diff[name] = diff
                sitk.WriteImage(diff, os.path.join(result_dir, ('diff_' + name + '%05d.vtk' % itr)))

        print('[done]')
