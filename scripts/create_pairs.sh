#!/usr/bin/env bash

root="/media/WDportable/MotionModelling_PMB"
cd /home/alina/Projects/GPR/scripts/

# ----------------------------------------------------------------------------------------------------------------------
# List of datasets
# Declare a string array with type
declare -a dataArray
declare -a nTrain
declare -a nVal
declare -a nTest
split_factor=11

dataArray[0]="114CTarchive_data15_16"
nTrain[0]=492  # in number of images
nVal[0]=55
nTest[0]=210
offset[0]=9
mode[0]=1

dataArray[1]="114CTarchive_EK-194-18_Geneva"
nTrain[1]=425 # in number of images
nVal[1]=55
nTest[1]=210
offset[1]=12
mode[1]=1

dataArray[2]="114CTarchive_vol01"
nTrain[2]=63 # in number of US videos
nVal[2]=5
nTest[2]=26
offset[2]=0
mode[2]=2

dataArray[3]="114CTarchive_vol04"
nTrain[3]=65 # in number of US videos
nVal[3]=5
nTest[3]=26
offset[3]=0
mode[3]=2

dataArray[4]="114CTarchive_vol05"
nTrain[4]=62 # in number of US videos
nVal[4]=5
nTest[4]=26
offset[4]=0
mode[4]=2

dataArray[5]="ZL0001_data15_16"
nTrain[5]=643  # in number of images
nVal[5]=55
nTest[5]=59
offset[5]=9
mode[5]=1

dataArray[6]="ZL0001_EK-194-18_Geneva"
nTrain[6]=576 # in number of images
nVal[6]=55
nTest[6]=59
offset[6]=12
mode[6]=1

dataArray[7]="ZL0001_vol01"
nTrain[7]=81 # in number of US videos
nVal[7]=5
nTest[7]=8
offset[7]=0
mode[7]=2

dataArray[8]="ZL0001_vol04"
nTrain[8]=83 # in number of US videos
nVal[8]=5
nTest[8]=8
offset[8]=0
mode[8]=2

dataArray[9]="ZL0001_vol05"
nTrain[9]=80 # in number of US videos
nVal[9]=5
nTest[9]=8
offset[9]=0
mode[9]=2


# ----------------------------------------------------------------------------------------------------------------------
# Preparation
# 1. Extract <data_folder.zip> with 4DCT(MRI) data from PSI
for data in "${dataArray[@]}"; do
  python data/create_CT_datasets.py --dir "$root/$data"
done

# ----------------------------------------------------------------------------------------------------------------------
# Create input (US)/output (4DCT(MRI)) pairs. Two possible scenarios:
# 1.  Stacking
# 2.  MoCo

# ----------------------------------------------------------------------------------------------------------------------
# 1.  Stacking
# 1a. for each dataset, run ~/Projects/4dmri/4dmri/ultrasound/helperFunctions/establishTempCorrespondence.m
# 1b. run ~/Projects/4dmri/4dmri/ultrasound/helperFunctions/extractCorrespondingUSImagesPMB.m
# 1c. Copy US data and list of pairs in appropriate folder
us_subdir="pairs/US"
rm -r "$root/${dataArray[0]}/$us_subdir"
rm -r "$root/${dataArray[5]}/$us_subdir"
rm -r "$root/${dataArray[1]}/$us_subdir"
rm -r "$root/${dataArray[6]}/$us_subdir"

cp -R "$root/US_data/dataG15_16/." "$root/${dataArray[0]}/$us_subdir"
cp -R "$root/US_data/dataG15_16/." "$root/${dataArray[5]}/$us_subdir"
cp -R "$root/US_data/dataG06_07/." "$root/${dataArray[1]}/$us_subdir"
cp -R "$root/US_data/dataG06_07/." "$root/${dataArray[6]}/$us_subdir"

cp "$root/US_data/pairs_dataG15_16.csv" "$root/${dataArray[0]}/pairs/pairs.csv"
cp "$root/US_data/pairs_dataG15_16.csv" "$root/${dataArray[5]}/pairs/pairs.csv"
cp "$root/US_data/pairs_dataG06_07.csv" "$root/${dataArray[1]}/pairs/pairs.csv"
cp "$root/US_data/pairs_dataG06_07.csv" "$root/${dataArray[6]}/pairs/pairs.csv"

# ----------------------------------------------------------------------------------------------------------------------
# 2.  MoCo:
# 2a. for each dataset, run ~/Projects/MRReconstruction/scripts/ReconstructionUS/preprocessing/data_loader.py

cd /home/alina/Projects/MRReconstruction/

dataArray[2]="114CTarchive_vol01"
dataArray[7]="ZL0001_vol01"
for i in 2 7
do
    python scripts/ReconstructionUS/preprocessing/data_loader.py \
            --us_root "$root/US_data/volunteer1_video120-213"  \
            --mr_root "$root/Moco_PMB/volunteer01/LungenAufnahmeVolunteer11/mvmt/displacements_inhalation" \
            --dest "$root/${dataArray[$i]}/pairs" \
            --split 0 0 ${nTrain[$i]} ${nVal[$i]} ${nTest[$i]}
done

dataArray[3]="114CTarchive_vol04"
dataArray[8]="ZL0001_vol04"
for i in 3 8
do
    python scripts/ReconstructionUS/preprocessing/data_loader.py \
            --us_root "$root/US_data/volunteer4_video051-146"  \
            --mr_root "$root/Moco_PMB/volunteer04/LungenAufnahmeVolunteer16/mvmt/displacements_inhalation" \
            --dest "$root/${dataArray[$i]}/pairs" \
            --split 0 0 ${nTrain[$i]} ${nVal[$i]} ${nTest[$i]}
done

dataArray[4]="114CTarchive_vol05"
dataArray[9]="ZL0001_vol05"
for i in 4 9
do
python scripts/ReconstructionUS/preprocessing/data_loader.py \
            --us_root "$root/US_data/volunteer5_video013-105"  \
            --mr_root "$root/Moco_PMB/volunteer05/LungenAufnahmeVolunteer14/mvmt/displacements_inhalation" \
            --dest "$root/${dataArray[$i]}/pairs" \
            --split 0 0 ${nTrain[$i]} ${nVal[$i]} ${nTest[$i]}
done

cd /home/alina/Projects/GPR/scripts/

# ----------------------------------------------------------------------------------------------------------------------
# 3. Prepare 4DCT(MRI) data
for i in 0 1 5
do
    python data/create_pairs.py \
            --root "/media/WDportable/MotionModelling_PMB/${dataArray[$i]}" \
            --split ${nTrain[$i]} ${nVal[$i]} ${nTest[$i]} \
            --offset ${offset[$i]} --mode ${mode[$i]}
done

for i in 6
do
    python data/create_pairs.py \
            --root "/media/WDportable/MotionModelling_PMB/${dataArray[$i]}" \
            --split ${nTrain[$i]} ${nVal[$i]} ${nTest[$i]} \
            --offset ${offset[$i]} --mode ${mode[$i]} \
            --ct_filename deformationfield_ref001_{:03d}.mha
done

for i in 2 3 4 7 8 9
do
    python data/create_pairs.py \
            --root "/media/WDportable/MotionModelling_PMB/${dataArray[$i]}" \
            --split $((${nTrain[$i]}*split_factor)) $((${nVal[$i]}*split_factor)) $((${nTest[$i]}*split_factor)) \
            --offset ${offset[$i]} --mode ${mode[$i]} \
            --ct_filename deformationfield_{:04d}.mha \
            --us_filename us_{:06d}.png
done

# ----------------------------------------------------------------------------------------------------------------------
# Then proceed with main.py (define new config file for each case)