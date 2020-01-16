Prepare data for 2 possible scenarios:
1. Slice Stacking (citation needed)
2. MoCo (citation needed)

For all datasets:
1. Extract <data_folder.zip> with 4DCT(MRI) data as received from PSI

Stacking:
2. For each dataset, run ~/Projects/4dmri/4dmri/ultrasound/helperFunctions/establishTempCorrespondence.m

Create Filestructure
python3 create_filestructure.py --config <filestructure_config.yaml>

Compute model
python3 main.py --config <params/config_dataset.yaml>
