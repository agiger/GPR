import os
import shutil as sh
import yaml
import argparse

from data import create_CT_datasets
from data import create_pairs

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='path to filestructure_XX.yaml file', type=str, required=True)
args = parser.parse_args()

if __name__ == "__main__":
    with open(args.config, 'r') as config_stream:
        cfg = yaml.safe_load(config_stream)

    opt = cfg['options']
    matlab_params = cfg['matlab_params']
    datasets = cfg['datasets']
    assert isinstance(datasets, object)

    # Create CT filestructure
    if opt['prepare_CT']:
        for data in datasets:
            print(data)
            ct_src = os.path.join(cfg['dest'], data)
            ct_dest = os.path.join(ct_src, 'pairs', 'CT')
            create_CT_datasets.create_datasets(ct_src, ct_dest, fmt='mha', tresh=30)

    # Create US filestructure
    if opt['prepare_US']:
        src = os.path.join(cfg['src'], 'US_data')
        us_root = os.path.join(cfg['dest'], 'US_data')

        # Run Matlab script (for stacking data)
        matlab_cmd = cfg['get_pairs_stacking'] + "('{:s}','{:s}',{:d},{:d},{:d},{:d},{:d})".format(
            src, us_root,
            matlab_params['crop_us_roi'], matlab_params['adjust_us_intensity'],
            matlab_params['n'], matlab_params['p'], matlab_params['us_framerate']
        )
        os.chdir(cfg['matlab'])
        os.system('matlab -nodisplay -nosplash -nodesktop -r "{:s};exit;"'.format(matlab_cmd))
        # os.system('matlab -nodisplay -nosplash -nodesktop -r "run({:s});exit;"'.format(cfg['get_pairs_stacking']))
        os.chdir(cfg['gpr'])

        # Copy US data
        for data in datasets:
            if datasets[data]['mode'] == 1:  # Slice stacking
                us_dest = os.path.join(cfg['dest'], data, 'pairs', 'US')
                if 'data15_16' in data:
                    us_src = os.path.join(us_root, 'dataG15_16')
                    us_pairs_lst = [os.path.join(us_root, f) for f in os.listdir(us_root)
                                    if f.startswith('pairs_dataG15_16') and f.endswith('csv')]
                elif 'EK-194-18' in data:
                    us_src = os.path.join(us_root, 'dataG06_07')
                    us_pairs_lst = [os.path.join(us_root, f) for f in os.listdir(us_root)
                                    if f.startswith('pairs_dataG06_07') and f.endswith('csv')]
                else:
                    raise Exception('US data for dataset {:s} not properly defined'.format(data))

                if os.path.exists(us_dest):
                    sh.rmtree(us_dest)
                sh.copytree(us_src, us_dest)
                pairs_dest = os.path.join(cfg['dest'], data, 'pairs')
                sh.copy2(us_pairs_lst[0], os.path.join(pairs_dest, 'pairs.csv'))

            elif datasets[data]['mode'] == 2:  # MoCo
                if 'vol01' in data:
                    us_root = os.path.join(cfg['src'], 'US_data/volunteer1_video120-213')
                    mr_root = os.path.join(cfg['src'], 'Moco_PMB/volunteer01/LungenAufnahmeVolunteer11/mvmt/displacements_inhalation')
                elif 'vol04' in data:
                    us_root = os.path.join(cfg['src'], 'US_data/volunteer4_video051-146')
                    mr_root = os.path.join(cfg['src'], 'Moco_PMB/volunteer04/LungenAufnahmeVolunteer16/mvmt/displacements_inhalation')
                elif 'vol05' in data:
                    us_root = os.path.join(cfg['src'], 'US_data/volunteer5_video013-105')
                    mr_root = os.path.join(cfg['src'], 'Moco_PMB/volunteer05/LungenAufnahmeVolunteer14/mvmt/displacements_inhalation')
                else:
                    raise Exception('US and MR data for dataset {:s} not properly defined'.format(data))

                cmd = []
                cmd.append('python3')
                cmd.append(cfg['get_pairs_moco'])
                cmd.append('--us_root {:s}'.format(us_root))
                cmd.append('--mr_root {:s}'.format(mr_root))
                cmd.append('--dest {:s}'.format(os.path.join(cfg['dest'], data, 'pairs')))
                cmd.append('--split {:d} {:d} {:d} {:d} {:d}'.format(
                    datasets[data]['nTrainAR'], datasets[data]['nTestAR'],
                    datasets[data]['nTrain'], datasets[data]['nVal'], datasets[data]['nTest']))

                if opt['autoregression']:
                    cmd.append('--ar')

                os.chdir(cfg['moco'])
                os.system(' '.join(cmd))
                os.chdir(cfg['gpr'])
            else:
                raise Exception('mode for dataset {:s} not correctly defined'.format(data))

    # Split US and CT data into training, validation and test sets
    if opt['create_pairs']:
        for data in datasets:
            root = os.path.join(cfg['dest'], data)
            if opt['autoregression']:
                split = [datasets[data]['nTrainAR'], datasets[data]['nTestAR'],
                         datasets[data]['nTrain'], datasets[data]['nVal'], datasets[data]['nTest']]
            else:
                split = [datasets[data]['nTrain'], datasets[data]['nVal'], datasets[data]['nTest']]
            split_factor = datasets[data]['split_factor']
            offset = datasets[data]['offset']
            mode = datasets[data]['mode']
            ct_filename = datasets[data]['ct_filename']
            us_filename = datasets[data]['us_filename']

            create_pairs.create_pairs(root, split, split_factor, offset, mode, opt['autoregression'], ct_filename, us_filename)



