# Title

## Introduction

### Requirements
This codebase is tested with:  
1.  h5py==3.1.0
2.  joypy==0.2.5
3.  matplotlib==3.4.2
4.  numpy==1.21.0
5.  pandas==1.2.3
6.  Pillow==8.4.0
7.  scikit_learn==1.0.1
8.  scipy==1.6.0
9.  seaborn==0.11.2
10. torch==1.8.1
11. torchvision==0.9.1
12. tqdm==4.60.0

To install all requirements, use "pip install -r requirements.txt"

## Running Experiments 
### Dataset Preparation
**MiniImageNet and CD-FSL:** Download the datasets for CD-FSL benchmark following step 1 and step 2 here: https://github.com/IBM/cdfsl-benchmark

**ImageNet:** https://www.kaggle.com/c/imagenet-object-localization-challenge/data

**Set datasets path:** Set the appropriate dataset pathes in "configs.py".

**Source dataset names:** "ImageNet", "miniImageNet"

**Target dataset names:** "EuroSAT", "CropDisease", "ChestX", "ISIC"

**All the dataset train/validation split files located at "datasets/split_seed_1" directory**

**All Baseline (miniImageNet) are trained Using an adapted version of the "https://github.com/cpphoo/STARTUP" repository**


| Near Domain (Table 1). BN: --model resnet10 and remove BTrans from the log path, VDT: --model resnet10_BTrans|                                                                                                                                                                                                   
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
python finetune.py --save_dir ./logs/baseline_teacher/BTrans/ImageNet_novel --target_dataset ImageNet_test --subset_split datasets/split_seed_1/ImageNet_val_labeled.csv --embedding_load_path ./logs/baseline_teacher/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans
  |
 [Pre-Trained Dictionary](./logs/baseline_teacher/checkpoint_best.pkl)                                                                                                          |

|CDFSL pure tuning on miniImageNet source  (Table 2). BN: --model resnet10 and remove BTrans from the log path, VDT: --model resnet10_BTrans. For Predict-BN comment the forward in Batchtransfer_EMA.py and comment out the Predict-BN forward function|                                                                                                                                                                                                   
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
python finetune.py --save_dir ./logs/baseline_teacher/BTrans/EuroSAT --target_dataset EuroSAT --subset_split datasets/split_seed_1/EuroSAT_labeled_80.csv --embedding_load_path ./logs/baseline_teacher/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans 
python finetune.py --save_dir ./logs/baseline_teacher/BTrans/CropDisease --target_dataset CropDisease --subset_split datasets/split_seed_1/CropDisease_labeled_80.csv --embedding_load_path ./logs/baseline_teacher/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans 
python finetune.py --save_dir ./logs/baseline_teacher/BTrans/ISIC --target_dataset ISIC --subset_split datasets/split_seed_1/ISIC_labeled_80.csv --embedding_load_path ./logs/baseline_teacher/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans 
python finetune.py --save_dir ./logs/baseline_teacher/BTrans/ChestX --target_dataset ChestX --subset_split datasets/split_seed_1/ChestX_labeled_80.csv --embedding_load_path ./logs/baseline_teacher/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans
  |
 [Pre-Trained Dictionary](./logs/baseline_teacher/checkpoint_best.pkl)                                                                                                          |
 

|CDFSL STARTUP on miniImageNet source  (Table 2). BN: --model resnet10 and remove BTrans from the log path, VDT: --model resnet10_BTrans| 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
python finetune.py --save_dir ./logs/STARTUP/BTrans/EuroSAT --target_dataset EuroSAT --subset_split datasets/split_seed_1/EuroSAT_labeled_80.csv --embedding_load_path ./logs/STARTUP/EuroSAT/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans 
[Pre-Trained Dictionary](./logs/STARTUP/EuroSAT/checkpoint_best.pkl)
python finetune.py --save_dir ./logs/STARTUP/BTrans/CropDisease --target_dataset CropDisease --subset_split datasets/split_seed_1/CropDisease_labeled_80.csv --embedding_load_path ./logs/STARTUP/CropDisease/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans 
[Pre-Trained Dictionary](./logs/STARTUP/CropDisease/checkpoint_best.pkl)
python finetune.py --save_dir ./logs/STARTUP/BTrans/ISIC --target_dataset ISIC --subset_split datasets/split_seed_1/ISIC_labeled_80.csv --embedding_load_path ./logs/STARTUP/ISIC/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans 
[Pre-Trained Dictionary](./logs/STARTUP/ISIC/checkpoint_best.pkl)
python finetune.py --save_dir ./logs/STARTUP/BTrans/ChestX --target_dataset ChestX --subset_split datasets/split_seed_1/ChestX_labeled_80.csv --embedding_load_path ./logs/STARTUP/ChestX/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans
[Pre-Trained Dictionary](./logs/STARTUP/ChestX/checkpoint_best.pkl)|

|CDFSL AdaBN on miniImageNet source  (Table 2). | 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
for target_testset in "ChestX" "ISIC" "EuroSAT" "CropDisease" do     python AdaBN.py --dir ./logs/AdaBN_miniImageNet/$target_testset --base_dictionary logs/baseline_teacher/checkpoint_best.pkl --target_dataset $target_testset --target_subset_split datasets/split_seed_1/$target_testset\_unlabeled_20.csv --bsize 256 --epochs 10 --model resnet18 done
python finetune.py --save_dir ./logs/AdaBN_miniImageNet/BTrans/EuroSAT --target_dataset EuroSAT --subset_split datasets/split_seed_1/EuroSAT_labeled_80.csv --embedding_load_path ./logs/AdaBN_miniImageNet/EuroSAT/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans 
[Pre-Trained Dictionary](./logs/AdaBN_miniImageNet/EuroSAT/checkpoint_best.pkl)
python finetune.py --save_dir ./logs/AdaBN_miniImageNet/BTrans/CropDisease --target_dataset CropDisease --subset_split datasets/split_seed_1/CropDisease_labeled_80.csv --embedding_load_path ./logs/AdaBN_miniImageNet/CropDisease/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans 
[Pre-Trained Dictionary](./logs/AdaBN_miniImageNet/CropDisease/checkpoint_best.pkl)
python finetune.py --save_dir ./logs/AdaBN_miniImageNet/BTrans/ISIC --target_dataset ISIC --subset_split datasets/split_seed_1/ISIC_labeled_80.csv --embedding_load_path ./logs/AdaBN_miniImageNet/ISIC/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans 
[Pre-Trained Dictionary](./logs/AdaBN_miniImageNet/ISIC/checkpoint_best.pkl)
python finetune.py --save_dir ./logs/AdaBN_miniImageNet/BTrans/ChestX --target_dataset ChestX --subset_split datasets/split_seed_1/ChestX_labeled_80.csv --embedding_load_path ./logs/AdaBN_miniImageNet/ChestX/checkpoint_best.pkl --freeze_backbone --model resnet10_BTrans
[Pre-Trained Dictionary](./logs/AdaBN_miniImageNet/ChestX/checkpoint_best.pkl)|


|CDFSL pure tuning on ImageNet source  (Table 3). BN: --model resnet18 and remove BTrans from the log path, VDT: --model resnet18_BTrans. For Predict-BN comment the forward in Batchtransfer_EMA.py and comment out the Predict-BN forward function|                                                                                                                                                                                                   
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
python ImageNet_finetune.py --save_dir ./logs/ImageNet/BTrans/EuroSAT --target_dataset EuroSAT --subset_split datasets/split_seed_1/EuroSAT_labeled_80.csv --embedding_load_path ./logs/ImageNet/checkpoint_best.pkl --freeze_backbone --model resnet18_BTrans
python ImageNet_finetune.py --save_dir ./logs/ImageNet/BTrans/CropDisease --target_dataset CropDisease --subset_split datasets/split_seed_1/CropDisease_labeled_80.csv --embedding_load_path ./logs/ImageNet/checkpoint_best.pkl --freeze_backbone --model resnet18_BTrans
python ImageNet_finetune.py --save_dir ./logs/ImageNet/BTrans/ISIC --target_dataset ISIC --subset_split datasets/split_seed_1/ISIC_labeled_80.csv --embedding_load_path ./logs/ImageNet/checkpoint_best.pkl --freeze_backbone --model resnet18_BTrans
python ImageNet_finetune.py --save_dir ./logs/ImageNet/BTrans/ChestX --target_dataset ChestX --subset_split datasets/split_seed_1/ChestX_labeled_80.csv --embedding_load_path ./logs/ImageNet/checkpoint_best.pkl --freeze_backbone --model resnet18_BTrans
  |
 [Pre-Trained Dictionary](./logs/ImageNet/checkpoint_best.pkl)                                                                                                          |

 |CDFSL AdaBN on ImageNet source  (Table 3). |                                                                                                                                                                                                   
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
for target_testset in "ChestX" "ISIC" "EuroSAT" "CropDisease" do     python AdaBN.py --dir ./logs/AdaBN_ImageNet/$target_testset --base_dictionary logs/baseline_teacher/checkpoint_best.pkl --target_dataset $target_testset --target_subset_split datasets/split_seed_1/$target_testset\_unlabeled_20.csv --bsize 256 --epochs 10 --model resnet18 done
python ImageNet_finetune.py --save_dir ./logs/AdaBN_ImageNet/EuroSAT --target_dataset EuroSAT --subset_split datasets/split_seed_1/EuroSAT_labeled_80.csv --embedding_load_path ./logs/AdaBN_ImageNet/EuroSAT/checkpoint_best.pkl --freeze_backbone --model resnet18
[Pre-Trained Dictionary](./logs/AdaBN_ImageNet/EuroSAT/checkpoint_best.pkl)
python ImageNet_finetune.py --save_dir ./logs/AdaBN_ImageNet/CropDisease --target_dataset CropDisease --subset_split datasets/split_seed_1/CropDisease_labeled_80.csv --embedding_load_path ./logs/AdaBN_ImageNet/CropDisease/checkpoint_best.pkl --freeze_backbone --model resnet18
[Pre-Trained Dictionary](./logs/AdaBN_ImageNet/CropDisease/checkpoint_best.pkl)
python ImageNet_finetune.py --save_dir ./logs/AdaBN_ImageNet/ISIC --target_dataset ISIC --subset_split datasets/split_seed_1/ISIC_labeled_80.csv --embedding_load_path ./logs/AdaBN_ImageNet/ISIC/checkpoint_best.pkl --freeze_backbone --model resnet18
[Pre-Trained Dictionary](./logs/AdaBN_ImageNet/ISIC/checkpoint_best.pkl)
python ImageNet_finetune.py --save_dir ./logs/AdaBN_ImageNet/ChestX --target_dataset ChestX --subset_split datasets/split_seed_1/ChestX_labeled_80.csv --embedding_load_path ./logs/AdaBN_ImageNet/ChestX/checkpoint_best.pkl --freeze_backbone --model resnet18
[Pre-Trained Dictionary](./logs/AdaBN_ImageNet/ChestX/checkpoint_best.pkl)


**In order to reproduce the plots, run the ./lab/{experiment}/plot.py**