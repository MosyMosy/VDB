from collections import OrderedDict
from math import trunc
import math
import statistics
import joypy
import matplotlib
import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
from pandas.core import groupby
import torch
import torch.nn as nn
from datasets import (Chest_few_shot, CropDisease_few_shot, EuroSAT_few_shot,
                      ISIC_few_shot, miniImageNet_few_shot)
from torchvision.models import resnet18
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy.stats.stats import mode
import copy
from lab.layers.BN_p import BatchNorm2d_p, BatchTransNorm_p
import gc

from PIL import Image


def load_checkpoint(model, load_path, device):
    '''
    Load model and optimizer from load path 
    Return the epoch to continue the checkpoint
    '''
    state = torch.load(load_path, map_location=torch.device(device))['state']
    clf_state = OrderedDict()
    state_keys = list(state.keys())
    for _, key in enumerate(state_keys):
        if "feature." in key:
            # an architecture model has attribute 'feature', load architecture
            # feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
            newkey = key.replace("feature.", "")
            state[newkey] = state.pop(key)
        elif "classifier." in key:
            newkey = key.replace("classifier.", "")
            clf_state[newkey] = state.pop(key)
        else:
            state.pop(key)
    model.load_state_dict(state)
    model.eval()
    return model


def load_checkpoint2(model, load_path, device):
    '''
    Load model and optimizer from load path 
    Return the epoch to continue the checkpoint
    '''
    sd = torch.load(load_path, map_location=torch.device(device))
    model.load_state_dict(sd['model'])
    model.eval()
    return model


def load_checkpoint3(model, load_path, device):
    '''
    Load model and optimizer from load path 
    Return the epoch to continue the checkpoint
    '''
    sd = torch.load(load_path, map_location=torch.device(device))
    # model = nn.DataParallel(model)

    model.load_state_dict(sd['state_dict'])
    model.eval()
    return model


def get_BN_output(model, colors, layers=None, channels=None, position='output', flatten=False):
    newcolors = []
    labels = []
    BN_list = []
    if (layers is None) or flatten:
        flatten = True
    else:
        flatten = False

    i = 0
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            if (layers is None) or (i in layers):
                flat_list = []
                out = layer.output.clone()
                out = out.permute(1, 0, 2, 3)
                out = out.flatten(start_dim=1).squeeze().tolist()
                for j, channel in enumerate(out):
                    if (channels is None) or (j in channels):
                        if flatten:
                            flat_list += channel
                        else:
                            flat_list.append(channel)

                if flatten:
                    BN_list.append(flat_list)
                    labels += ['Layer {0:02d} ({1: 0.2f}, {2: 0.2f})'.format(
                        i+1, torch.tensor(flat_list).mean(), torch.tensor(flat_list).std())]
                else:
                    BN_list += flat_list
                    if (channels is not None) and (len(channels) == 1):
                        labels += ['Layer {0:02d} ({1: 0.2f}, {2: 0.2f})'.format(
                            i+1, statistics.mean(flat_list[0]), statistics.stdev(flat_list[0]))]
                    else:
                        labels += ['Layer {0:02d}'.format(i+1)]
                    if channels is None:
                        labels += [None]*(len(out)-1)
                    else:
                        labels += [None]*(len(channels)-1)

                    clm = LinearSegmentedColormap.from_list(
                        "Custom", colors, N=len(out))
                    temp = clm(range(0, len(out)))
                    for c in temp:
                        newcolors.append(c)

            i += 1
    if flatten:
        clm = LinearSegmentedColormap.from_list(
            "Custom", colors, N=i)
        temp = clm(range(0, i))
        for c in temp:
            newcolors.append(c)

    return BN_list, labels, ListedColormap(newcolors, name='custom')


def compare_domains(base_x, EuroSAT_x, color_range, layers=[[None]], channels=None):
    model_BN = resnet18(pretrained=True, norm_layer=BatchNorm2d_p)
    model_BTrans = resnet18(pretrained=True, norm_layer=BatchTransNorm_p)

    for l in layers:
        path_list = []

        with torch.no_grad():
            model_BN(base_x)
            mini_out, mini_labels, mini_clm = get_BN_output(
                model_BN, colors=color_range[0], layers=l, channels=channels)

            model_BN(EuroSAT_x)
            Euro_out, EuroSAT_labels, EuroSAT_clm = get_BN_output(
                model_BN, colors=color_range[1], layers=l, channels=channels)

            model_BTrans(EuroSAT_x)
            Euro_out_BTrans, EuroSAT_labels_BTrans, EuroSAT_clm_BTrans = get_BN_output(
                model_BTrans, colors=color_range[3], layers=l, channels=channels)

            out_BN, labels_BN, clm_BN = copy.copy(
                mini_out), copy.copy(mini_labels), copy.copy(mini_clm)
            out_BN += list(reversed(Euro_out))
            labels_BN += [None] * len(EuroSAT_labels)
            clm_BN = list(clm_BN.colors)
            clm_BN += list(EuroSAT_clm.colors)
            clm_BN = ListedColormap(clm_BN, name='custom')

            out_BTrans, labels_BTrans, clm_BTrans = copy.copy(
                mini_out), copy.copy(mini_labels), copy.copy(mini_clm)
            out_BTrans += list(reversed(Euro_out_BTrans))
            labels_BTrans += [None] * len(EuroSAT_labels_BTrans)
            clm_BTrans = list(clm_BTrans.colors)
            clm_BTrans += list(EuroSAT_clm_BTrans.colors)
            clm_BTrans = ListedColormap(clm_BTrans, name='custom')

            y_up = 10
            x_range = [-2, 2]
            if l in [[0],[1],[2],[14],[17],[7]]:
                y_up = 15
            elif l in [[12]]:
                y_up = 35
            
            if l in [[19]]:
                x_range = [-10, 10]
            args = {'overlap': 4, 'bw_method': 0.2,
                    'linewidth': 0.3, 'linecolor': 'w',
                    'x_range': x_range, 'ylim': [0, y_up], 'alpha': 0.6, 'figsize': (10, 5), 'fill': True,
                    'grid': False, 'kind': 'kde', 'hist': False, 'bins': int(len(base_x))}

            joypy.joyplot(out_BN, labels=list(
                reversed(labels_BN)), colormap=clm_BN, **args)
            # plt.show()
            path_list.append(
                "./lab/layers/Resnet18_with_BN_layer-{0}.png".format(l))
            plt.savefig(path_list[-1], transparent=True, dpi=1200)

            joypy.joyplot(out_BTrans, labels=list(
                reversed(labels_BTrans)), colormap=clm_BTrans, **args)
            # plt.show()
            path_list.append(
                "./lab/layers/Resnet18_with_BTrans_layer-{0}.png".format(l))
            plt.savefig(path_list[-1], transparent=True, dpi=1200)

            plt.close("all")
            gc.collect()
            print(l)

device = torch.device("cpu")

b_size = 32
transform = EuroSAT_few_shot.TransformLoader(
    224).get_composed_transform(aug=True)
transform_test = EuroSAT_few_shot.TransformLoader(
    224).get_composed_transform(aug=False)
split = 'datasets/split_seed_1/EuroSAT_unlabeled_20.csv'
dataset = EuroSAT_few_shot.SimpleDataset(
    transform, split=split)
EuroSAT_loader = torch.utils.data.DataLoader(dataset, batch_size=b_size,
                                             num_workers=0,
                                             shuffle=True, drop_last=True)

transform = miniImageNet_few_shot.TransformLoader(
    224).get_composed_transform(aug=True)
transform_test = miniImageNet_few_shot.TransformLoader(
    224).get_composed_transform(aug=False)
dataset = miniImageNet_few_shot.SimpleDataset(
    transform, split=None)
base_loader = torch.utils.data.DataLoader(dataset, batch_size=b_size,
                                          num_workers=0,
                                          shuffle=True, drop_last=True)

EuroSAT_x, _ = iter(EuroSAT_loader).next()
base_x, _ = iter(base_loader).next()


color_range = [['#670022', '#FF6699'], ['#004668', '#66D2FF'],
               ['#9B2802', '#FF9966'], ['#346600', '#75E600']]

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
layers = [[7]]  # [None] is for full network
channels = None

compare_domains(base_x=base_x, EuroSAT_x=EuroSAT_x, color_range=color_range,
                layers=layers, channels=channels)

