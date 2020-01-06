#!/usr/bin/env bash

train_size=220
#splits=(000 110 220 330 440 550 660)
splits=(000 110 220 330 440)

#data_root="/media/WDportable/MotionModelling_PMB/ZL0001_vol05/pairs"
#mask="/media/WDportable/MotionModelling_PMB/ZL0001_Vol5_VOI.mha"
data_root="/media/WDportable/MotionModelling_PMB/114CTarchive_vol05/pairs"
mask="/media/WDportable/MotionModelling_PMB/114CTarchive_Vol5_VOI.mha"

build="/home/alina/Projects/GPR/build"
cd "$build"
#make -j12

#for i in "${splits[@]}"
#do
#    gpr_dir="$data_root/CT/gpr_drift-$i-$train_size"
#    mkdir $gpr_dir
#
#    result_dir="$data_root/CT/validation_pred_drift-$i-$train_size"
#    mkdir $result_dir

#    # gpLearn
#    ./apps/gpLearn \
#        "$data_root/US/train" \
#        "$data_root/CT/train" \
#        "GaussianKernel(35, 30,)" 1.0 \
#        "$gpr_dir/gpr" \
#        64 26 $i $train_size

    # gpPredict
#    ./apps/gpPredict \
#        "$gpr_dir/gpr" \
#        "$data_root/US/validation" \
#        "$result_dir" \
#        "$data_root/CT/validation" \
#        "$data_root/CT/train/deformationfield_0000.mha" \
#        64 26
#done

# Validation
cd ../scripts

for i in "${splits[@]}"
do
    python validation_dvf.py --root "$data_root/CT" --suffix "_drift-$i-$train_size" --noshow --mask "$mask"
done

# Complete training set
python validation_dvf.py --root "$data_root/CT" --noshow --mask "$mask"
