#!/usr/bin/env bash

#python main.py --config params/config_ZL0001_data15_16.yaml
#python main.py --config params/config_ZL0001_EK-194-18_Geneva.yaml
#python main.py --config params/config_ZL0001_vol01.yaml
#python main.py --config params/config_ZL0001_vol04.yaml
#python main.py --config params/config_ZL0001_vol05.yaml

#python main.py --config params/config_114CTarchive_data15_16.yaml
#python main.py --config params/config_114CTarchive_EK-194-18_Geneva.yaml
#python main.py --config params/config_114CTarchive_vol01.yaml
#python main.py --config params/config_114CTarchive_vol04.yaml
#python main.py --config params/config_114CTarchive_vol05.yaml

root="/media/WDportable/MotionModelling_PMB"
#python vtk_mha_converter.py -src "$root/ZL0001_data15_16/pairs/CT/test_pred" -dest "$root/ZL0001_data15_16/pairs/CT/test_pred"
#python vtk_mha_converter.py -src "$root/ZL0001_EK-194-18_Geneva/pairs/CT/test_pred" -dest "$root/ZL0001_EK-194-18_Geneva/pairs/CT/test_pred"
#python vtk_mha_converter.py -src "$root/ZL0001_vol01/pairs/CT/test_pred" -dest "$root/ZL0001_vol01/pairs/CT/test_pred"
#python vtk_mha_converter.py -src "$root/ZL0001_vol04/pairs/CT/test_pred" -dest "$root/ZL0001_vol04/pairs/CT/test_pred"
#python vtk_mha_converter.py -src "$root/ZL0001_vol05/pairs/CT/test_pred" -dest "$root/ZL0001_vol05/pairs/CT/test_pred"

python vtk_mha_converter.py -src "$root/114CTarchive_data15_16/pairs/CT/test_pred" -dest "$root/114CTarchive_data15_16/pairs/CT/test_pred_mha"
python vtk_mha_converter.py -src "$root/114CTarchive_EK-194-18_Geneva/pairs/CT/test_pred" -dest "$root/114CTarchive_EK-194-18_Geneva/pairs/CT/test_pred_mha"
python vtk_mha_converter.py -src "$root/114CTarchive_vol01/pairs/CT/test_pred" -dest "$root/114CTarchive_vol01/pairs/CT/test_pred_mha"
python vtk_mha_converter.py -src "$root/114CTarchive_vol04/pairs/CT/test_pred" -dest "$root/114CTarchive_vol04/pairs/CT/test_pred_mha"
python vtk_mha_converter.py -src "$root/114CTarchive_vol05/pairs/CT/test_pred" -dest "$root/114CTarchive_vol05/pairs/CT/test_pred_mha"
