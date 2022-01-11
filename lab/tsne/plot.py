from collections import OrderedDict

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from seaborn.palettes import color_palette
import numpy as np

# import seaborn as sns
import torch
from torch import nn
from torchvision.models import resnet18
import os

from Batchtransfer_EMA import BatchTransNorm

from datasets import (Chest_few_shot, CropDisease_few_shot, EuroSAT_few_shot,
                      ISIC_few_shot, miniImageNet_few_shot)


def get_feature_list(model_BN, model_VDT, source_loader, target_loder, dataset_names_list = ["ImageNet","EuroSAT"]):    
    label_dataset = []
    feature_list = []
    
    for x, _ in source_loader:            
        feature_list += model_BN(x)            
        label_dataset += [dataset_names_list[0]]*len(x)
    
    for x, _ in target_loder:            
        feature_list += model_BN(x)            
        label_dataset += [dataset_names_list[1]+ "_BN"]*len(x)   
    
    for x, _ in target_loder:            
        feature_list += model_VDT(x)            
        label_dataset += [dataset_names_list[1] + "_VDT"]*len(x)     
    
    return torch.stack(feature_list).numpy(), label_dataset


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

b_size = 128

source_dataset = miniImageNet_few_shot
transform = source_dataset.TransformLoader(
        224).get_composed_transform(aug=True)
transform_test = source_dataset.TransformLoader(
    224).get_composed_transform(aug=False)
# split = 'datasets/split_seed_1/{0}_labeled_20.csv'.format(
#     dataset_names_list[i])
# if dataset_names_list[i] == 'miniImageNet':
split = None
dataset = source_dataset.SimpleDataset(
    transform, split=split)
source_dataloader = torch.utils.data.DataLoader(dataset, batch_size=b_size,
                                        num_workers=0,
                                        shuffle=True, drop_last=True)
source_x, _ = iter(source_dataloader).next()

target_dataset = EuroSAT_few_shot
transform = target_dataset.TransformLoader(
        224).get_composed_transform(aug=True)
transform_test = target_dataset.TransformLoader(
    224).get_composed_transform(aug=False)
# split = 'datasets/split_seed_1/{0}_labeled_20.csv'.format(
#     dataset_names_list[i])
# if dataset_names_list[i] == 'miniImageNet':
split = None
dataset = target_dataset.SimpleDataset(
    transform, split=split)
target_dataloader = torch.utils.data.DataLoader(dataset, batch_size=b_size,
                                        num_workers=0,
                                        shuffle=True, drop_last=True)
target_x, _ = iter(target_dataloader).next()

model_BN = resnet18(pretrained=True)
model_BN.eval()
model_BN.fc = nn.Identity(512)
model_VDT = resnet18(pretrained=True, norm_layer=BatchTransNorm)
model_VDT.eval()
model_VDT.fc = nn.Identity(512)

label_dataset = []
feature_list = []

dataset_names_list = ["ImageNet","EuroSAT"]

          
feature_list += model_BN(source_x)            
label_dataset += [dataset_names_list[0]]*len(source_x)
            
feature_list += model_BN(target_x)            
label_dataset += [dataset_names_list[1]+ "_BN"]*len(target_x)   
        
feature_list += model_VDT(target_x)            
label_dataset += [dataset_names_list[1] + "_VDT"]*len(target_x)     

baseline_features, labels = torch.stack(feature_list).detach().numpy(), label_dataset

# baseline_features, labels = get_feature_list(model_BN, model_VDT, source_dataloader, target_dataloader)
color = sns.color_palette(n_colors=3)
perplexitys = range(0, 101, 2)
for perplexity in perplexitys:
    fig = plt.figure(figsize=(10, 10))    
    
    baseline_embedding = TSNE().fit_transform(baseline_features)    
    sns.kdeplot(x=baseline_embedding[:, 0], y=baseline_embedding[:, 1],
                hue=labels, palette=color)
    
    # baseline_na_embedding = TSNE(perplexity=perplexity, n_iter=5000, learning_rate=10).fit_transform(baseline_na_features)    
    # sns.kdeplot(x=baseline_na_embedding[:, 0], y=baseline_na_embedding[:, 1],
    #             hue=labels, ax=ax[1], palette=color)
    title = 'Left baseline, Right baseline_na. Perplexity {0}'.format(perplexity)
    fig.suptitle(title)
    
    plt.savefig('./lab/tsne/result/default_{0}.png'.format(title))
    print(title)
    break
