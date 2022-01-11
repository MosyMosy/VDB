#!/bin/bash
#SBATCH --mail-user=Moslem.Yazdanpanah@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=Fine-Tune_BTrans
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=0-06:00
#SBATCH --account=rrg-ebrahimi

nvidia-smi

source ~/ENV/bin/activate

echo "------------------------------------< Data preparation>----------------------------------"
echo "Copying the source code"
date +"%T"
cd $SLURM_TMPDIR
cp -r ~/scratch/StylishTENT .

echo "Copying the datasets"
date +"%T"
cp -r ~/scratch/CD-FSL_Datasets .

echo "creating data directories"
date +"%T"
cd StylishTENT
cd BMS
cd data
unzip -q $SLURM_TMPDIR/CD-FSL_Datasets/miniImagenet.zip
unzip -q $SLURM_TMPDIR/CD-FSL_Datasets/ILSVRC_val.zip

mkdir ChestX-Ray8 EuroSAT ISIC2018 plant-disease

cd EuroSAT
unzip ~/scratch/CD-FSL_Datasets/EuroSAT.zip
cd ..

cd ChestX-Ray8
unzip ~/scratch/CD-FSL_Datasets/ChestX-Ray8.zip
mkdir images
find . -type f -name '*.png' -print0 | xargs -0 mv -t images
cd ..

cd ISIC2018
unzip ~/scratch/CD-FSL_Datasets/ISIC2018.zip
unzip ~/scratch/CD-FSL_Datasets/ISIC2018_GroundTruth.zip
cd ..

cd plant-disease
unzip ~/scratch/CD-FSL_Datasets/plant-disease.zip

echo "----------------------------------< End of data preparation>--------------------------------"
date +"%T"
echo "--------------------------------------------------------------------------------------------"

echo "---------------------------------------<Run the program>------------------------------------"
date +"%T"
cd $SLURM_TMPDIR

cd StylishTENT
cd BMS
 
# ------------------------------------------ baseline mini -----------------------------------
python finetune.py --save_dir ./logs/baseline_teacher/BTrans/EuroSAT --target_dataset EuroSAT --subset_split datasets/split_seed_1/EuroSAT_labeled_80.csv --embedding_load_path ./logs/baseline_teacher/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans &
python finetune.py --save_dir ./logs/baseline_teacher/BTrans/CropDisease --target_dataset CropDisease --subset_split datasets/split_seed_1/CropDisease_labeled_80.csv --embedding_load_path ./logs/baseline_teacher/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans &
python finetune.py --save_dir ./logs/baseline_teacher/BTrans/ISIC --target_dataset ISIC --subset_split datasets/split_seed_1/ISIC_labeled_80.csv --embedding_load_path ./logs/baseline_teacher/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans &
wait
python finetune.py --save_dir ./logs/baseline_teacher/BTrans/ChestX --target_dataset ChestX --subset_split datasets/split_seed_1/ChestX_labeled_80.csv --embedding_load_path ./logs/baseline_teacher/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans
python finetune.py --save_dir ./logs/baseline_teacher/BTrans/ImageNet_novel --target_dataset ImageNet_test --subset_split datasets/split_seed_1/ImageNet_val_labeled.csv --embedding_load_path ./logs/baseline_teacher/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans


# ------------------------------------------ baseline_na mini -----------------------------------
python finetune.py --save_dir ./logs/baseline_na_teacher/BTrans/EuroSAT --target_dataset EuroSAT --subset_split datasets/split_seed_1/EuroSAT_labeled_80.csv --embedding_load_path ./logs/baseline_na_teacher/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans &
python finetune.py --save_dir ./logs/baseline_na_teacher/BTrans/CropDisease --target_dataset CropDisease --subset_split datasets/split_seed_1/CropDisease_labeled_80.csv --embedding_load_path ./logs/baseline_na_teacher/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans &
python finetune.py --save_dir ./logs/baseline_na_teacher/BTrans/ISIC --target_dataset ISIC --subset_split datasets/split_seed_1/ISIC_labeled_80.csv --embedding_load_path ./logs/baseline_na_teacher/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans &
wait
python finetune.py --save_dir ./logs/baseline_na_teacher/BTrans/ChestX --target_dataset ChestX --subset_split datasets/split_seed_1/ChestX_labeled_80.csv --embedding_load_path ./logs/baseline_na_teacher/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans
python finetune.py --save_dir ./logs/baseline_na_teacher/BTrans/ImageNet_novel --target_dataset ImageNet_test --subset_split datasets/split_seed_1/ImageNet_val_labeled.csv --embedding_load_path ./logs/baseline_na_teacher/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans


# ------------------------------------------ STARTUP mini -----------------------------------
python finetune.py --save_dir ./logs/STARTUP/BTrans/EuroSAT --target_dataset EuroSAT --subset_split datasets/split_seed_1/EuroSAT_labeled_80.csv --embedding_load_path ./logs/STARTUP/EuroSAT/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans &
python finetune.py --save_dir ./logs/STARTUP/BTrans/CropDisease --target_dataset CropDisease --subset_split datasets/split_seed_1/CropDisease_labeled_80.csv --embedding_load_path ./logs/STARTUP/CropDisease/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans &
python finetune.py --save_dir ./logs/STARTUP/BTrans/ISIC --target_dataset ISIC --subset_split datasets/split_seed_1/ISIC_labeled_80.csv --embedding_load_path ./logs/STARTUP/ISIC/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans &
wait
python finetune.py --save_dir ./logs/STARTUP/BTrans/ChestX --target_dataset ChestX --subset_split datasets/split_seed_1/ChestX_labeled_80.csv --embedding_load_path ./logs/STARTUP/ChestX/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans


# ------------------------------------------ STARTUP_na_tf mini -----------------------------------
python finetune.py --save_dir ./logs/STARTUP_na_tf/BTrans/EuroSAT --target_dataset EuroSAT --subset_split datasets/split_seed_1/EuroSAT_labeled_80.csv --embedding_load_path ./logs/STARTUP_na_tf/EuroSAT/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans &
python finetune.py --save_dir ./logs/STARTUP_na_tf/BTrans/CropDisease --target_dataset CropDisease --subset_split datasets/split_seed_1/CropDisease_labeled_80.csv --embedding_load_path ./logs/STARTUP_na_tf/CropDisease/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans &
python finetune.py --save_dir ./logs/STARTUP_na_tf/BTrans/ISIC --target_dataset ISIC --subset_split datasets/split_seed_1/ISIC_labeled_80.csv --embedding_load_path ./logs/STARTUP_na_tf/ISIC/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans &
wait
python finetune.py --save_dir ./logs/STARTUP_na_tf/BTrans/ChestX --target_dataset ChestX --subset_split datasets/split_seed_1/ChestX_labeled_80.csv --embedding_load_path ./logs/STARTUP_na_tf/ChestX/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans


# ------------------------------------------ baseline ImageNet -----------------------------------
python ImageNet_finetune.py --save_dir ./logs/ImageNet/BTrans/EuroSAT --target_dataset EuroSAT --subset_split datasets/split_seed_1/EuroSAT_labeled_80.csv --embedding_load_path ./logs/ImageNet/checkpoint_best.pkl --freeze_backbone --model resnet18_BTrans
python ImageNet_finetune.py --save_dir ./logs/ImageNet/BTrans/CropDisease --target_dataset CropDisease --subset_split datasets/split_seed_1/CropDisease_labeled_80.csv --embedding_load_path ./logs/ImageNet/checkpoint_best.pkl --freeze_backbone --model resnet18_BTrans
python ImageNet_finetune.py --save_dir ./logs/ImageNet/BTrans/ISIC --target_dataset ISIC --subset_split datasets/split_seed_1/ISIC_labeled_80.csv --embedding_load_path ./logs/ImageNet/checkpoint_best.pkl --freeze_backbone --model resnet18_BTrans
wait
python ImageNet_finetune.py --save_dir ./logs/ImageNet/BTrans/ChestX --target_dataset ChestX --subset_split datasets/split_seed_1/ChestX_labeled_80.csv --embedding_load_path ./logs/ImageNet/checkpoint_best.pkl --freeze_backbone --model resnet18_BTrans

# ------------------------------------------ baseline_na ImageNet -----------------------------------
python ImageNet_finetune.py --save_dir ./logs/ImageNet_na/BTrans/EuroSAT --target_dataset EuroSAT --subset_split datasets/split_seed_1/EuroSAT_labeled_80.csv --embedding_load_path ./logs/ImageNet_na/checkpoint_best.pkl --freeze_backbone --model resnet18_BTrans
python ImageNet_finetune.py --save_dir ./logs/ImageNet_na/BTrans/CropDisease --target_dataset CropDisease --subset_split datasets/split_seed_1/CropDisease_labeled_80.csv --embedding_load_path ./logs/ImageNet_na/checkpoint_best.pkl --freeze_backbone --model resnet18_BTrans
python ImageNet_finetune.py --save_dir ./logs/ImageNet_na/BTrans/ISIC --target_dataset ISIC --subset_split datasets/split_seed_1/ISIC_labeled_80.csv --embedding_load_path ./logs/ImageNet_na/checkpoint_best.pkl --freeze_backbone --model resnet18_BTrans
wait
python ImageNet_finetune.py --save_dir ./logs/ImageNet_na/BTrans/ChestX --target_dataset ChestX --subset_split datasets/split_seed_1/ChestX_labeled_80.csv --embedding_load_path ./logs/ImageNet_na/checkpoint_best.pkl --freeze_backbone --model resnet18_BTrans


# ------------------------------------------ AdaBN -----------------------------------
for target_testset in "ChestX" "ISIC" "EuroSAT" "CropDisease"
do
    python AdaBN.py --dir ./logs/AdaBN_ImageNet/$target_testset --base_dictionary logs/baseline_teacher/checkpoint_best.pkl --target_dataset $target_testset --target_subset_split datasets/split_seed_1/$target_testset\_unlabeled_20.csv --bsize 256 --epochs 10 --model resnet18
done
wait

python ImageNet_finetune.py --save_dir ./logs/AdaBN_ImageNet/EuroSAT --target_dataset EuroSAT --subset_split datasets/split_seed_1/EuroSAT_labeled_80.csv --embedding_load_path ./logs/AdaBN_ImageNet/EuroSAT/checkpoint_best.pkl --freeze_backbone --model resnet18
python ImageNet_finetune.py --save_dir ./logs/AdaBN_ImageNet/CropDisease --target_dataset CropDisease --subset_split datasets/split_seed_1/CropDisease_labeled_80.csv --embedding_load_path ./logs/AdaBN_ImageNet/CropDisease/checkpoint_best.pkl --freeze_backbone --model resnet18
python ImageNet_finetune.py --save_dir ./logs/AdaBN_ImageNet/ISIC --target_dataset ISIC --subset_split datasets/split_seed_1/ISIC_labeled_80.csv --embedding_load_path ./logs/AdaBN_ImageNet/ISIC/checkpoint_best.pkl --freeze_backbone --model resnet18
python ImageNet_finetune.py --save_dir ./logs/AdaBN_ImageNet/ChestX --target_dataset ChestX --subset_split datasets/split_seed_1/ChestX_labeled_80.csv --embedding_load_path ./logs/AdaBN_ImageNet/ChestX/checkpoint_best.pkl --freeze_backbone --model resnet18

python ImageNet_finetune.py --save_dir ./logs/AdaBN_ImageNet_na/EuroSAT --target_dataset EuroSAT --subset_split datasets/split_seed_1/EuroSAT_labeled_80.csv --embedding_load_path ./logs/AdaBN_ImageNet_na/EuroSAT/checkpoint_best.pkl --freeze_backbone --model resnet18
python ImageNet_finetune.py --save_dir ./logs/AdaBN_ImageNet_na/CropDisease --target_dataset CropDisease --subset_split datasets/split_seed_1/CropDisease_labeled_80.csv --embedding_load_path ./logs/AdaBN_ImageNet_na/CropDisease/checkpoint_best.pkl --freeze_backbone --model resnet18
python ImageNet_finetune.py --save_dir ./logs/AdaBN_ImageNet_na/ISIC --target_dataset ISIC --subset_split datasets/split_seed_1/ISIC_labeled_80.csv --embedding_load_path ./logs/AdaBN_ImageNet_na/ISIC/checkpoint_best.pkl --freeze_backbone --model resnet18
python ImageNet_finetune.py --save_dir ./logs/AdaBN_ImageNet_na/ChestX --target_dataset ChestX --subset_split datasets/split_seed_1/ChestX_labeled_80.csv --embedding_load_path ./logs/AdaBN_ImageNet_na/ChestX/checkpoint_best.pkl --freeze_backbone --model resnet18

python finetune.py --save_dir ./logs/AdaBN_STARTUP/EuroSAT --target_dataset EuroSAT --subset_split datasets/split_seed_1/EuroSAT_labeled_80.csv --embedding_load_path ./logs/AdaBN_STARTUP/EuroSAT/checkpoint_best.pkl --freeze_backbone --model resnet10
python finetune.py --save_dir ./logs/AdaBN_STARTUP/CropDisease --target_dataset CropDisease --subset_split datasets/split_seed_1/CropDisease_labeled_80.csv --embedding_load_path ./logs/AdaBN_STARTUP/CropDisease/checkpoint_best.pkl --freeze_backbone --model resnet10
python finetune.py --save_dir ./logs/AdaBN_STARTUP/ISIC --target_dataset ISIC --subset_split datasets/split_seed_1/ISIC_labeled_80.csv --embedding_load_path ./logs/AdaBN_STARTUP/ISIC/checkpoint_best.pkl --freeze_backbone --model resnet10
python finetune.py --save_dir ./logs/AdaBN_STARTUP/ChestX --target_dataset ChestX --subset_split datasets/split_seed_1/ChestX_labeled_80.csv --embedding_load_path ./logs/AdaBN_STARTUP/ChestX/checkpoint_best.pkl --freeze_backbone --model resnet10

python finetune.py --save_dir ./logs/AdaBN_STARTUP_na_tf/EuroSAT --target_dataset EuroSAT --subset_split datasets/split_seed_1/EuroSAT_labeled_80.csv --embedding_load_path ./logs/AdaBN_STARTUP_na_tf/EuroSAT/checkpoint_best.pkl --freeze_backbone --model resnet10
 python finetune.py --save_dir ./logs/AdaBN_STARTUP_na_tf/ChestX --target_dataset ChestX --subset_split datasets/split_seed_1/ChestX_labeled_80.csv --embedding_load_path ./logs/AdaBN_STARTUP_na_tf/ChestX/checkpoint_best.pkl --freeze_backbone --model resnet10


echo "-----------------------------------<End of run the program>---------------------------------"
date +"%T"
echo "--------------------------------------<backup the result>-----------------------------------"
date +"%T"
cd $SLURM_TMPDIR
cp -r $SLURM_TMPDIR/StylishTENT/BMS/logs/ ~/scratch/StylishTENT/BMS/
