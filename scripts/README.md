Prepare data for 2 possible scenarios:
1. Slice Stacking (citation needed)
2. MoCo (citation needed)

For all datasets:
1. Extract <data_folder.zip> with 4DCT(MRI) data as received from PSI

Stacking:
2. For each dataset, run ~/Projects/4dmri/4dmri/ultrasound/helperFunctions/establishTempCorrespondence.m

For tracking study
1. Create filestructure, compute model, predict results on validation data
python3 run_experiments.py --root /media/WDportable/MotionModelling_Tracking --config params/tracking/validation --config_filestructure filestructure_tracking.yaml

2. Predict test data, convert results to mha
python3 run_experiments.py --root /media/WDportable/MotionModelling_Tracking --config params/tracking/test --convert_vtk2mha
